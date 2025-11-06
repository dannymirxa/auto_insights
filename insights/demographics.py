from typing_extensions import List
import pandas as pd

class DemographicsInsights:
    def __init__(self, df: pd.DataFrame, demographic_cols: List[str], metric_columns: List[str], num_spots: int = 3):
        self.df = df
        self.demographic_cols = demographic_cols
        self.metric_columns = metric_columns
        self.output: dict[str, dict] = {}   # instance attribute
        self.num_spots = num_spots

    def _average_all_drivers_score_by_demographic(self) -> dict:
        for demographics_col in self.demographic_cols:
            df_long = self.df.melt(
                id_vars=demographics_col,
                value_vars=self.metric_columns,
                var_name="drivers",
                value_name="average_score"
            )
            df_grouped = df_long.groupby(demographics_col)["average_score"].mean().reset_index()
            self.output[f"average_all_drivers_score_by_demographic"] = {f"{demographics_col}": df_grouped.to_dict(orient='records')}

    def _drivers_scores_by_demographics_and_drivers(self) -> dict:
        for demographics_col in self.demographic_cols:
            df_grouped = (
                self.df.groupby(demographics_col)[self.metric_columns]
                .mean()
                .reset_index()
            )
            self.output[f"drivers_scores_by_demographics_and_drivers"] = {f"{demographics_col}": df_grouped.to_dict(orient='records')}

    def _brightspots_hotspots_average_all_drivers_score_by_demographic(self) -> dict:
        for demographics_col in self.demographic_cols:
            df_long = self.df.melt(
                id_vars=demographics_col,
                value_vars=self.metric_columns,
                var_name="drivers",
                value_name="average_score"
            )
            df_grouped = df_long.groupby(demographics_col)["average_score"].mean()
            top = df_grouped.nlargest(self.num_spots)
            best_demographics = [
                    {"demographic": demographic, "score": float(score)}
                    for demographic, score in top.items()
                ]
            bottom = df_grouped.nsmallest(self.num_spots)
            worst_demographics = [
                    {"demographic": demographic, "score": float(score)}
                    for demographic, score in bottom.items()
                ]
            self.output.setdefault(
                    "brightspots_hotspots_average_all_drivers_score_by_demographic", []
                    ).append({
                        "demographic": demographics_col,
                        f"best_{demographics_col}": best_demographics,
                        f"worst_{demographics_col}": worst_demographics,
                    })
            
    def _brightspots_hotspots_drivers_scores_by_demographics_and_drivers(self):
         for demographics_col in self.demographic_cols:
            df_grouped = (
                self.df.groupby(demographics_col)[self.metric_columns]
                .mean()
            )
            for group, row in df_grouped.iterrows():
                top = row.nlargest(self.num_spots)
                bottom = row.nsmallest(self.num_spots)
                best_drivers = [
                    {"driver": driver, "score": float(score)}
                    for driver, score in top.items()
                ]
                worst_drivers = [
                    {"driver": driver, "score": float(score)}
                    for driver, score in bottom.items()
                ]
                self.output.setdefault(
                    "brightspots_hotspots_drivers_scores_by_demographics_and_drivers", []
                    ).append({
                        "demographic": demographics_col,
                        "group": group,
                        "best_drivers": best_drivers,
                        "worst_drivers": worst_drivers,
                    })
                
    def _correlation_by_drivers(self) -> dict:
        df_grouped = self.df[self.metric_columns]
        df = df_grouped.corr().reset_index()
        self.output.setdefault(
            "_correlation_by_drivers", []
            ).append({
                "correlation_matrix": df.to_dict(orient='records')
            })

    def _correlation_by_demographics_agg_scores(self) -> dict:
        for demographics_col in self.demographic_cols:
            df_grouped = (
                self.df.groupby(demographics_col)[self.metric_columns]
                .mean()
            )
            # self.output[f"drivers_scores_by_demographics_and_drivers"] = {f"{demographics_col}": df_grouped.to_dict(orient='records')}
            df = df_grouped.corr().reset_index()
            self.output.setdefault(
                "correlation_by_demographics_agg_scores", []
                ).append({
                    "demographic": demographics_col,
                    "correlation_matrix": df.to_dict(orient='records')
                })


    def get_output(self) -> dict[str, dict]:
        self._brightspots_hotspots_average_all_drivers_score_by_demographic()
        self._brightspots_hotspots_drivers_scores_by_demographics_and_drivers()
        self._correlation_by_drivers()
        self._correlation_by_demographics_agg_scores()
        return self.output