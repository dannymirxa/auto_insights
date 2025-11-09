from typing_extensions import List
import pandas as pd

class TransformationDriverInsights:
    def __init__(self, df: pd.DataFrame, demographic_cols: List[str], df_map: pd.DataFrame, num_spots: int = 3):
        self.df = df.copy()
        self.demographic_cols = demographic_cols
        self.df_qcode_map = df_map.copy()
        self.num_spots = num_spots
        
        self.output: dict[str, dict] = {}

        self.qcodes = self.df_qcode_map["qcode"].unique().tolist()
        self.essential_columns = self.demographic_cols + self.qcodes

        # Validate mapping
        if 'qcode' not in self.df_qcode_map.columns or 'driver' not in self.df_qcode_map.columns:
            raise ValueError("df_qcode_map must contain 'qcode' and 'driver' columns")

        self.df_melted_by_qcode = self.df[self.essential_columns].melt(
                            id_vars= self.demographic_cols,
                            value_vars= self.qcodes, 
                            var_name='qcode',
                            value_name='score'
                        )
        
        self.df_unpivot = pd.merge(self.df_melted_by_qcode, self.df_qcode_map, left_on="qcode", right_on="qcode", how="inner")
        self.df_unpivot['score_percent'] = ((self.df_unpivot["score"] - 1) / (7 -1)) * 100

    def _average_all_qcodes_score_by_driver(self) -> None:
        df_grouped = self.df_unpivot.groupby('driver')["score"].mean()
        self.output[f"average_all_qcodes_score_by_driver"] = df_grouped.to_dict()

    def _average_all_qcodes_score_by_qcode(self) -> None:
        df_grouped = self.df_unpivot.groupby('qcode')["score"].mean()
        self.output[f"average_all_qcodes_score_by_qcode"] = df_grouped.to_dict()

    def _brightspots_hotspots_average_all_qcodes_score_by_driver(self) -> None:
        df_grouped = self.df_unpivot.groupby('driver')["score"].mean()
        top = df_grouped.nlargest(self.num_spots)
        bottom = df_grouped.nsmallest(self.num_spots)

        self.output.setdefault(
            "brightspots_hotspots_average_all_qcodes_score_by_driver", []
            ).append({
                "best_drivers": top.to_dict(),
                "worst_drivers": bottom.to_dict(),
            })
    
    def _brightspots_hotspots_average_all_score_by_driver_and_qcodes(self) -> None:
        qcode_avg = self.df_unpivot.groupby(["driver", "qcode"], as_index=False)["score"].mean()
        driver_avg = qcode_avg.groupby("driver", as_index=False)["score"].mean().rename(columns={"score": "driver_avg"})

        merged = qcode_avg.merge(driver_avg, on="driver")
        sorted_df = merged.sort_values(by=["driver_avg", "score"], ascending=[False, False])
        
        result = {}
        for driver, group in sorted_df.groupby("driver"):
            result[driver] = dict(zip(group["qcode"], group["score"]))
        self.output.setdefault(
            "brightspots_hotspots_average_all_score_by_driver_and_qcodes", []
            ).append(result)
        
    def get_output(self) -> dict[str, dict]:
        """
        Run the full driver-centric analysis pipeline and return the combined output.
        """
        self._average_all_qcodes_score_by_driver()
        self._average_all_qcodes_score_by_qcode()
        self._brightspots_hotspots_average_all_qcodes_score_by_driver()
        self._brightspots_hotspots_average_all_score_by_driver_and_qcodes()
        return self.output