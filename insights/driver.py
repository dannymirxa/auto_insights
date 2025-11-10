from typing_extensions import List, Optional
import pandas as pd
import numpy as np

class TransformationDriverInsights:
    def __init__(self, df_new: pd.DataFrame, demographic_cols: List[str], df_map: pd.DataFrame, df_old: Optional[pd.DataFrame] = None, num_spots: int = 3):
        self.df_new = df_new.copy()
        self.df_old = df_old.copy() if isinstance(df_old, pd.DataFrame) else None
        self.demographic_cols = demographic_cols
        self.df_qcode_map = df_map.copy()
        self.num_spots = num_spots
        
        self.output: dict[str, dict] = {}

        self.qcodes = self.df_qcode_map["qcode"].unique().tolist()
        self.essential_columns = self.demographic_cols + self.qcodes

        # Validate mapping
        if 'qcode' not in self.df_qcode_map.columns or 'driver' not in self.df_qcode_map.columns:
            raise ValueError("df_qcode_map must contain 'qcode' and 'driver' columns")

        # Build unified unpivot with Survey indicator: 1=new, 2=old
        self.has_old = False
        df_new_melt = self.df_new[self.essential_columns].melt(
            id_vars=self.demographic_cols,
            value_vars=self.qcodes,
            var_name="qcode",
            value_name="score",
        )
        df_new_melt["Survey"] = 1

        df_melt = df_new_melt
        # Include df_old if provided and structurally valid
        if isinstance(self.df_old, pd.DataFrame) and all(c in self.df_old.columns for c in self.essential_columns):
            df_old_melt = self.df_old[self.essential_columns].melt(
                id_vars=self.demographic_cols,
                value_vars=self.qcodes,
                var_name="qcode",
                value_name="score",
            )
            df_old_melt["Survey"] = 2
            df_melt = pd.concat([df_new_melt, df_old_melt], ignore_index=True)
            self.has_old = True

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
        if "Survey" in df.columns:
            df = df[df["Survey"] == 1]
        df_grouped = df.groupby("driver")["score"].mean()
        self.output["average_all_qcodes_score_by_driver"] = df_grouped.to_dict()

    def _average_all_qcodes_score_by_qcode(self) -> None:
        df = self.df_unpivot.copy()
        if "Survey" in df.columns:
            df = df[df["Survey"] == 1]
        df_grouped = df.groupby("qcode")["score"].mean()
        self.output["average_all_qcodes_score_by_qcode"] = df_grouped.to_dict()

    def _brightspots_hotspots_average_all_qcodes_score_by_driver(self) -> None:
        df = self.df_unpivot.copy()
        if "Survey" in df.columns:
            df = df[df["Survey"] == 1]
        df_grouped = df.groupby("driver")["score"].mean()
        top = df_grouped.nlargest(self.num_spots)
        bottom = df_grouped.nsmallest(self.num_spots)

        self.output.setdefault(
            "brightspots_hotspots_average_all_qcodes_score_by_driver", []
        ).append({
            "best_drivers": top.to_dict(),
            "worst_drivers": bottom.to_dict(),
        })
    
    def _brightspots_hotspots_average_all_score_by_driver_and_qcodes(self) -> None:
        df = self.df_unpivot.copy()
        if "Survey" in df.columns:
            df = df[df["Survey"] == 1]
        qcode_avg = df.groupby(["driver", "qcode"], as_index=False)["score"].mean()
        driver_avg = qcode_avg.groupby("driver", as_index=False)["score"].mean().rename(columns={"score": "driver_avg"})

        merged = qcode_avg.merge(driver_avg, on="driver")
        sorted_df = merged.sort_values(by=["driver_avg", "score"], ascending=[False, False])
        
        result = {}
        for driver, group in sorted_df.groupby("driver"):
            result[driver] = dict(zip(group["qcode"], group["score"]))
        self.output.setdefault(
            "brightspots_hotspots_average_all_score_by_driver_and_qcodes", []
        ).append(result)

    def _score_zone_by_driver_and_qcode(self) -> None:
        """
        Compute per-row zone using row-level thresholds and aggregate into {driver: {qcode: zone}}.
        """
        df = self.df_unpivot.copy()
        if "Survey" in df.columns:
            df = df[df["Survey"] == 1]
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
        Compute driver averages, thresholds (mean per driver), employee confidence, rank, and reconciled status.
        """
        df = self.df_unpivot.copy()
        if "Survey" in df.columns:
            df = df[df["Survey"] == 1]
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

        driver_avg = df.groupby("driver", as_index=False)["score"].mean().rename(columns={"score": "driver_avg"})

        cols = ["off_track_percentile", "on_track_percentile", "high_performance_percentile"]
        if all(c in df.columns for c in cols):
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            thresh = df.groupby("driver", as_index=False)[cols].mean()
            merged = driver_avg.merge(thresh, on="driver", how="left")

            score = merged["driver_avg"]
            off = merged["off_track_percentile"]
            on = merged["on_track_percentile"]
            hi = merged["high_performance_percentile"]

            # Confidence buckets: mirrors cascade from transformation_summary
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

            # Dense rank: highest avg = 1
            merged["driver_rank"] = merged["driver_avg"].rank(ascending=False, method="dense").astype("Int64")

            # Base Status from rank
            merged["Status"] = np.where(merged["driver_rank"] <= 3, "top performing", "low performing")

            # Driver-level zone from driver_avg vs thresholds
            merged["score_zone"] = np.where(
                merged["driver_avg"] >= merged["high_performance_percentile"],
                "High Performance Zone",
                np.where(
                    merged["driver_avg"] >= merged["on_track_percentile"],
                    "On-Track Zone",
                    np.where(
                        merged["driver_avg"] >= merged["off_track_percentile"],
                        "Unsustainable Zone",
                        "Off-Track Zone",
                    ),
                ),
            )

            # Reconcile Status using zone context
            mask_hi_on = merged["score_zone"].isin(["High Performance Zone", "On-Track Zone"])
            merged.loc[mask_hi_on & (merged["Status"] == "low performing"), "Status"] = "top performing"

            mask_off_uns = merged["score_zone"].isin(["Unsustainable Zone", "Off-Track Zone"])
            merged.loc[mask_off_uns & (merged["Status"] == "top performing"), "Status"] = "ahead of other drivers but still poor performance"

            # Append outputs
            self.output["driver_average_score"] = dict(zip(merged["driver"], merged["driver_avg"]))
            self.output["driver_rank"] = {k: (int(v) if pd.notnull(v) else None) for k, v in dict(zip(merged["driver"], merged["driver_rank"])).items()}
            self.output["driver_employee_confidence_level"] = dict(zip(merged["driver"], merged["employee_confidence_level"]))
            self.output["driver_status"] = dict(zip(merged["driver"], merged["Status"]))
        else:
            # Fallback: no thresholds available; provide averages and ranks only
            ser_avg = driver_avg.set_index("driver")["driver_avg"]
            ser_rank = ser_avg.rank(ascending=False, method="dense").astype("Int64")
            self.output["driver_average_score"] = ser_avg.to_dict()
            self.output["driver_rank"] = {k: (int(v) if pd.notnull(v) else None) for k, v in ser_rank.to_dict().items()}

    def _score_diff_by_driver_qcode(self) -> None:
        """
        Compute new-old diffs per driver/qcode using Survey indicator (1=new, 2=old).
        """
        if not getattr(self, "has_old", False):
            return
        df = self.df_unpivot.copy()
        if "Survey" not in df.columns:
            return

        by_qcode = df.groupby(["driver", "qcode", "Survey"], as_index=False)["score"].mean()
        pivot_q = by_qcode.pivot_table(index=["driver", "qcode"], columns="Survey", values="score", aggfunc="mean")

        new_series = pivot_q[1] if 1 in pivot_q.columns else pd.Series(index=pivot_q.index, dtype=float)
        old_series = pivot_q[2] if 2 in pivot_q.columns else pd.Series(index=pivot_q.index, dtype=float)
        diff_q = (pd.to_numeric(new_series, errors="coerce") - pd.to_numeric(old_series, errors="coerce"))

        diff_map: dict[str, dict] = {}
        for (driver, qcode), val in diff_q.items():
            diff_map.setdefault(driver, {})[qcode] = float(val) if pd.notnull(val) else None

        self.output["score_diff_by_driver_qcode"] = diff_map

    def _score_diff_by_driver_avg(self) -> None:
        """
        Compute new-old diffs for driver-average scores using Survey indicator (1=new, 2=old).
        """
        if not getattr(self, "has_old", False):
            return
        df = self.df_unpivot.copy()
        if "Survey" not in df.columns:
            return

        by_driver = df.groupby(["driver", "Survey"], as_index=False)["score"].mean()
        pivot_d = by_driver.pivot_table(index=["driver"], columns="Survey", values="score", aggfunc="mean")

        new_d = pivot_d[1] if 1 in pivot_d.columns else pd.Series(index=pivot_d.index, dtype=float)
        old_d = pivot_d[2] if 2 in pivot_d.columns else pd.Series(index=pivot_d.index, dtype=float)
        diff_d = (pd.to_numeric(new_d, errors="coerce") - pd.to_numeric(old_d, errors="coerce"))

        self.output["score_diff_by_driver_avg"] = {k: (float(v) if pd.notnull(v) else None) for k, v in diff_d.items()}
        
    def get_output(self) -> dict[str, dict]:
        """
        Run the full driver-centric analysis pipeline and return the combined output.
        Uses new cycle (Survey==1) for primary insights; computes diffs if old cycle present.
        """
        self._average_all_qcodes_score_by_driver()
        self._average_all_qcodes_score_by_qcode()
        self._brightspots_hotspots_average_all_qcodes_score_by_driver()
        self._brightspots_hotspots_average_all_score_by_driver_and_qcodes()
        # Enrich with zones and confidence/status for single-cycle runs
        self._score_zone_by_driver_and_qcode()
        self._driver_rank_confidence_and_status()
        # If both cycles provided, add delta insights
        self._score_diff_by_driver_qcode()
        self._score_diff_by_driver_avg()
        return self.output