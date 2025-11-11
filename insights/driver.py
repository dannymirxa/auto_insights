from typing_extensions import List
import pandas as pd
import numpy as np

class TransformationDriverInsights:
    def __init__(self, df_new: pd.DataFrame, demographic_cols: List[str], df_map: pd.DataFrame, num_spots: int = 3):
        self.df_new = df_new.copy()
        self.demographic_cols = demographic_cols
        self.df_qcode_map = df_map.copy()
        self.num_spots = num_spots
        
        self.output: dict[str, dict] = {}

        self.qcodes = self.df_qcode_map["qcode"].unique().tolist()
        self.essential_columns = self.demographic_cols + self.qcodes

        # Validate mapping
        if 'qcode' not in self.df_qcode_map.columns or 'driver' not in self.df_qcode_map.columns:
            raise ValueError("df_qcode_map must contain 'qcode' and 'driver' columns")

        # Build unpivot from new data only
        df_melt = self.df_new[self.essential_columns].melt(
            id_vars=self.demographic_cols,
            value_vars=self.qcodes,
            var_name="qcode",
            value_name="score",
        )

        # Merge mapping
        self.df_unpivot = pd.merge(df_melt, self.df_qcode_map, left_on="qcode", right_on="qcode", how="inner")

        # Normalize threshold column names to lowercase with underscores if present
        rename_map = {
            "Off Track Percentile": "off_track_percentile",
            "On Track Percentile": "on_track_percentile",
            "High Performance Percentile": "high_performance_percentile",
        }
        present = {k: v for k, v in rename_map.items() if k in self.df_unpivot.columns}
        if present:
            self.df_unpivot.rename(columns=present, inplace=True)

        # Convert score to percentage (1–7 Likert to 0–100)
        self.df_unpivot["score"] = ((pd.to_numeric(self.df_unpivot["score"], errors="coerce") - 1) / (7 - 1)) * 100

    def _average_all_qcodes_score_by_driver(self) -> None:
        df = self.df_unpivot.copy()
        df_grouped = df.groupby("driver")["score"].mean()
        self.output["average_all_qcodes_score_by_driver"] = df_grouped.to_dict()

    def _average_all_qcodes_score_by_qcode(self) -> None:
        df = self.df_unpivot.copy()
        df_grouped = df.groupby("qcode")["score"].mean()
        self.output["average_all_qcodes_score_by_qcode"] = df_grouped.to_dict()

    def _compute_driver_rank_and_spots(self) -> tuple[pd.Series, pd.Series, dict, dict]:
        """
        Compute driver averages and ranks once, plus top/bottom N spot dictionaries.
        Returns: (driver_avg_series, driver_rank_series, best_dict, worst_dict)
        """
        df = self.df_unpivot.copy()
        driver_avg = df.groupby("driver")["score"].mean()
        # Cache for reuse
        self._driver_avg = driver_avg
        driver_rank = driver_avg.rank(ascending=False, method="dense").astype("Int64")
        self._driver_rank = driver_rank
        top = driver_avg.nlargest(self.num_spots)
        bottom = driver_avg.nsmallest(self.num_spots)
        return driver_avg, driver_rank, top.to_dict(), bottom.to_dict()

    def _brightspots_hotspots_average_all_qcodes_score_by_driver(self) -> None:
        """
        Use the shared computation to produce top/bottom N driver spots.
        """
        _, _, best, worst = self._compute_driver_rank_and_spots()
        self.output.setdefault(
            "brightspots_hotspots_average_all_qcodes_score_by_driver", []
        ).append({
            "best_drivers": best,
            "worst_drivers": worst,
        })
    
    def _brightspots_hotspots_average_all_score_by_driver_and_qcodes(self) -> None:
        """
        For each driver, pick top and bottom num_spots qcodes by average score.
        Output structure:
        { driver: { "best_qcode": {q1: score, ...}, "worst_qcode": {qN: score, ...} } }
        """
        df = self.df_unpivot.copy()
        qcode_avg = df.groupby(["driver", "qcode"], as_index=False)["score"].mean()

        result = {}
        for driver, group in qcode_avg.groupby("driver"):
            group_sorted_desc = group.sort_values("score", ascending=False)
            best = dict(zip(
                group_sorted_desc["qcode"].head(self.num_spots),
                group_sorted_desc["score"].head(self.num_spots),
            ))
            group_sorted_asc = group.sort_values("score", ascending=True)
            worst = dict(zip(
                group_sorted_asc["qcode"].head(self.num_spots),
                group_sorted_asc["score"].head(self.num_spots),
            ))
            result[driver] = {
                "best_qcode": best,
                "worst_qcode": worst,
            }
        self.output.setdefault(
            "brightspots_hotspots_average_all_score_by_driver_and_qcodes", []
        ).append(result)

    def _score_zone_by_driver_and_qcode(self) -> None:
        """
        Compute per-row zone using row-level thresholds and aggregate into {driver: {qcode: zone}}.
        """
        df = self.df_unpivot.copy()
        cols = ["off_track_percentile", "on_track_percentile", "high_performance_percentile"]
        if not all(c in df.columns for c in cols):
            return
        # Coerce thresholds and score to numeric
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Aggregate by (driver, qcode) to compute mean score and mean thresholds, then derive zone
        agg = df.groupby(["driver", "qcode"], as_index=False)[
            ["score", "off_track_percentile", "on_track_percentile", "high_performance_percentile"]
        ].mean()

        agg["score_zone"] = np.where(
            agg["score"] >= agg["high_performance_percentile"],
            "High Performance Zone",
            np.where(
                agg["score"] >= agg["on_track_percentile"],
                "On-Track Zone",
                np.where(
                    agg["score"] >= agg["off_track_percentile"],
                    "Unsustainable Zone",
                    "Off-Track Zone",
                ),
            ),
        )

        zone_map: dict[str, dict] = {}
        for driver, g in agg.groupby("driver"):
            zone_map[driver] = dict(zip(g["qcode"], g["score_zone"]))
        self.output["score_zone_by_driver_qcode"] = zone_map

    def _driver_rank_confidence_and_status(self) -> None:
        """
        Compute driver rank once and derive driver_status as:
        - top N => "top performing"
        - bottom N => "low performing"
        - middle => "average performing"
        Also compute employee_confidence_level from thresholds.
        """
        df = self.df_unpivot.copy()
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

        # Ensure shared averages and ranks are computed once
        driver_avg_series = getattr(self, "_driver_avg", None)
        driver_rank_series = getattr(self, "_driver_rank", None)
        if driver_avg_series is None or driver_rank_series is None:
            # Compute via shared helper for consistency with bright/hot
            driver_avg_series, driver_rank_series, top_map, bottom_map = self._compute_driver_rank_and_spots()
        else:
            # Build maps from cached series
            top_map = driver_avg_series.nlargest(self.num_spots).to_dict()
            bottom_map = driver_avg_series.nsmallest(self.num_spots).to_dict()

        # Output driver_rank from shared series
        self.output["driver_rank"] = {k: (int(v) if pd.notnull(v) else None) for k, v in driver_rank_series.to_dict().items()}

        # Employee confidence level from thresholds (if available)
        cols = ["off_track_percentile", "on_track_percentile", "high_performance_percentile"]
        if all(c in df.columns for c in cols):
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            # Aggregate thresholds per driver
            thresh = df.groupby("driver", as_index=False)[cols].mean()
            merged = pd.DataFrame({"driver": driver_avg_series.index, "driver_avg": driver_avg_series.values})
            merged = merged.merge(thresh, on="driver", how="left")

            score = merged["driver_avg"]
            off = merged["off_track_percentile"]
            on = merged["on_track_percentile"]
            hi = merged["high_performance_percentile"]

            merged["employee_confidence_level"] = np.select(
                [
                    score >= hi,
                    score >= (hi - 10),
                    score >= on,
                    score >= off,
                    score < off,
                ],
                ["Very Strong", "Strong", "Moderate", "Low", "Very Low"],
                default="Very Low",
            )
            self.output["driver_employee_confidence_level"] = dict(zip(merged["driver"], merged["employee_confidence_level"]))

        # Derive driver_status purely from rank-based extremes with neutral middle
        top_set = set(top_map.keys())
        bottom_set = set(bottom_map.keys())
        status_map = {}
        for driver in driver_avg_series.index:
            if driver in top_set:
                status_map[driver] = "top performing"
            elif driver in bottom_set:
                status_map[driver] = "low performing"
            else:
                status_map[driver] = "average performing"
        self.output["driver_status"] = status_map

    def get_output(self) -> dict[str, dict]:
        """
        Run the full driver-centric analysis pipeline and return the combined output.
        Insights are computed using the provided dataset only.
        """
        self._average_all_qcodes_score_by_driver()
        self._average_all_qcodes_score_by_qcode()
        self._brightspots_hotspots_average_all_qcodes_score_by_driver()
        self._brightspots_hotspots_average_all_score_by_driver_and_qcodes()
        # Enrich with zones and confidence/status
        self._score_zone_by_driver_and_qcode()
        self._driver_rank_confidence_and_status()
        return self.output