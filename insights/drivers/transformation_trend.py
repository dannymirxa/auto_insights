import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


class TransformationTrendSummary:
    """
    Stateful trend transformation summarizer. Holds train_data and survey.
    """

    def __init__(self, train_data: pd.DataFrame, survey: int):
        self.train_data = train_data
        self.survey = survey

    def build_prompt(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute full pipeline: convert data, prepare subset, and emit JSON records.
        """
        df = self.convert_data(self.train_data.copy())
        df_demo = self._prepare_for_json(df, self.survey)
        final_json = self._create_json(df_demo)
        return df_demo, final_json

    @staticmethod
    def _clean_empty(d):
        if isinstance(d, dict):
            return {k: v for k, v in ((k, TransformationTrendSummary._clean_empty(v)) for k, v in d.items()) if v}
        if isinstance(d, list):
            return [v for v in map(TransformationTrendSummary._clean_empty, d) if v]
        return d

    # ----------------------------
    # Data conversion (previously convert_data)
    # ----------------------------
    def convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates data transformations without changing underlying logic.
        """
        df = df.copy()

        # sort by survey before diff
        if "Survey" in df.columns:
            df = df.sort_values("Survey")

        # score diff per Qcode/Type
        group_cols = ["Qcode", "Type"] if "Qcode" in df.columns else ["qcode", "Type"]
        df.loc[:, "score_diff"] = df.groupby(group_cols)["Score"].diff(1)

        # driver ranks per survey
        df.loc[:, "driver_rank"] = df.groupby(["Type", "Survey"])["Score"].rank(ascending=False, method="dense").astype(
            "Int64"
        )
        df.loc[df.Type != "Driver", "driver_rank"] = np.nan

        # deduplicate
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        # confidence level for drivers (element-wise thresholds preserved)
        off_track_s = pd.to_numeric(df["Off Track Percentile"], errors="coerce") if "Off Track Percentile" in df.columns else pd.Series(np.nan, index=df.index)
        on_track_s = pd.to_numeric(df["On Track Percentile"], errors="coerce") if "On Track Percentile" in df.columns else pd.Series(np.nan, index=df.index)
        high_perf_s = pd.to_numeric(df["High Performance Percentile"], errors="coerce") if "High Performance Percentile" in df.columns else pd.Series(np.nan, index=df.index)

        df.loc[(df.Type == "Driver") & (df.Score >= off_track_s), "employee_confidence_level"] = "Low"
        df.loc[(df.Type == "Driver") & (df.Score >= on_track_s), "employee_confidence_level"] = "Moderate"  # bucket 1
        df.loc[(df.Type == "Driver") & (df.Score >= (high_perf_s - 10)), "employee_confidence_level"] = "Strong"  # bucket 2
        df.loc[(df.Type == "Driver") & (df.Score >= high_perf_s), "employee_confidence_level"] = "Very Strong"
        df.loc[(df.Type == "Driver") & (df.Score < off_track_s), "employee_confidence_level"] = "Very Low"

        # quantiles for score_diff (pandas 2 named agg)
        qdiff = self._compute_quantiles(df, metric="score_diff")
        df = pd.merge(df, qdiff, on=["Survey"])

        # outlier flag for score_diff outside [q1, q3]
        df.loc[:, "score_diff_flag"] = np.where((df.score_diff < df.q1) | (df.score_diff > df.q3), df.score_diff, np.nan)

        # sentiment for questions based on Score thresholds
        df.loc[(df.Type == "Question") & (df.Score >= on_track_s), "Employee Perception"] = "Positive"
        df.loc[(df.Type == "Question") & (df.Score >= high_perf_s), "Employee Perception"] = "Very Positive"
        df.loc[(df.Type == "Question") & (df.Score < on_track_s), "Employee Perception"] = "Negative"
        df.loc[(df.Type == "Question") & (df.Score < off_track_s), "Employee Perception"] = "Very Negative"

        # reversed scales: support both 'Reversed' and 'reversed'
        if "Reversed" in df.columns:
            rev = df["Reversed"].astype(bool)
        elif "reversed" in df.columns:
            rev = df["reversed"].astype(bool)
        else:
            rev = pd.Series(False, index=df.index)

        mask_q = df.Type == "Question"
        df.loc[mask_q & rev & (df["Employee Perception"] == "Positive"), "Employee Perception"] = "Negative"
        df.loc[mask_q & rev & (df["Employee Perception"] == "Very Positive"), "Employee Perception"] = "Very Negative"
        df.loc[mask_q & rev & (df["Employee Perception"] == "Negative"), "Employee Perception"] = "Positive"
        df.loc[mask_q & rev & (df["Employee Perception"] == "Very Negative"), "Employee Perception"] = "Very Positive"

        # replace 'number' -> 'level' for specific drivers
        if "Driver" in df.columns:
            mask_special = df["Driver"].isin(["Fear & Frustration", "Passion & Drive"])
            if "Question" in df.columns:
                df.loc[mask_special, "Question"] = df.loc[mask_special, "Question"].str.replace(
                    "number", "level", case=False, regex=False
                )
            if "Description" in df.columns:
                df.loc[mask_special, "Description"] = df.loc[mask_special, "Description"].str.replace(
                    "number", "level", case=False, regex=False
                )

        # question score_diff trends
        df.loc[(df.score_diff > 0) & (df.Type == "Question"), "change_from_last_cycle"] = "slightly better"
        df.loc[(df.score_diff >= 10) & (df.Type == "Question"), "change_from_last_cycle"] = "better"
        df.loc[(df.score_diff >= 20) & (df.Type == "Question"), "change_from_last_cycle"] = "significantly better"
        df.loc[(df.score_diff < 0) & (df.Type == "Question"), "change_from_last_cycle"] = "slightly worsened"
        df.loc[(df.score_diff < -10) & (df.Type == "Question"), "change_from_last_cycle"] = "worsened"
        df.loc[(df.score_diff < -20) & (df.Type == "Question"), "change_from_last_cycle"] = "significantly worsened"

        # driver score_diff trends
        col = "driver_performance_change_from_last_cycle"
        df.loc[(df.Type == "Driver") & (df.score_diff > 0), col] = "slightly better"
        df.loc[(df.Type == "Driver") & (df.score_diff >= 10), col] = "better"
        df.loc[(df.Type == "Driver") & (df.score_diff >= 16), col] = "significantly better"
        df.loc[(df.Type == "Driver") & (df.score_diff < 0), col] = "slightly worsened"
        df.loc[(df.Type == "Driver") & (df.score_diff < -10), col] = "worsened"
        df.loc[(df.Type == "Driver") & (df.score_diff < -20), col] = "significantly worsened"
        df.loc[(df.Type == "Driver") & (df.score_diff == 0), col] = "no change"

        # sanitize description
        if "Description" in df.columns:
            df["Description"] = df["Description"].str.replace("* ", "", case=False, regex=False)
            df["Description"] = df["Description"].str.replace("^ ", "", case=False, regex=False)

        # cleanup helper cols
        df = df.drop(columns=["q1", "q3"], errors="ignore")

        return df

    @staticmethod
    def _compute_quantiles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        agg = (
            df.groupby(["Survey"], dropna=False)[metric]
            .agg(q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
            .reset_index()
        )
        return agg

    # ----------------------------
    # JSON preparation (no data manipulation beyond structuring)
    # ----------------------------
    @staticmethod
    def _prepare_for_json(df: pd.DataFrame, survey: int) -> pd.DataFrame:
        col_names = [
            "Driver",
            "Type",
            "Description",
            "employee_confidence_level",
            "overall_change_from_last_cyle",
            "change_from_last_cycle",
            "Employee Perception",
            "driver_rank",
            "score_diff",
            "score_diff_flag",
            "driver_performance_change_from_last_cycle",
        ]

        df_demo = df[(df.Survey == survey)].reset_index(drop=True).drop_duplicates()

        if "Driver" in df_demo.columns:
            df_demo.loc[:, "Driver"] = df_demo["Driver"].str.lower()
        if "Description" in df_demo.columns:
            df_demo.loc[:, "Description"] = df_demo["Description"].str.lower()

        # filter relevant drivers: significant question movement
        if df_demo["score_diff"].notnull().all():
            drivers_ls = (
                df_demo[
                    (df_demo.Type == "Question")
                    & ((df_demo.score_diff_flag.notnull()) | (df_demo.score_diff.abs().gt(10)))
                ]
                .Driver.unique()
                .tolist()
            )
            df_demo = df_demo[df_demo.Driver.isin(drivers_ls)].reset_index(drop=True)

        # sort and prune
        df_demo = df_demo.sort_values(by=["driver_rank", "Driver"])
        df_demo = df_demo.dropna(
            subset=["score_diff", "change_from_last_cycle", "Employee Perception", "driver_performance_change_from_last_cycle"],
            how="all",
        )

        # rename and forward-fill group-wise fields
        df_demo = df_demo.rename(
            columns={"driver_performance_change_from_last_cycle": "overall_change_from_last_cyle"}
        )
        df_demo.loc[:, "overall_change_from_last_cyle"] = df_demo.groupby(["Driver"])["overall_change_from_last_cyle"].ffill()
        df_demo.loc[:, "employee_confidence_level"] = df_demo.groupby(["Driver"])["employee_confidence_level"].ffill()

        # retain only required columns for JSON creation
        keep_cols = [c for c in col_names if c in df_demo.columns]
        df_demo = df_demo[keep_cols]

        return df_demo

    @staticmethod
    def _create_json(df_demo: pd.DataFrame) -> List[Dict[str, Any]]:
        group_cols = ["driver_rank", "Driver", "overall_change_from_last_cyle", "employee_confidence_level"]
        present_group_cols = [c for c in group_cols if c in df_demo.columns]

        json_df = (
            df_demo.groupby(present_group_cols, dropna=False)
            .apply(lambda x: x[["Description", "Employee Perception", "change_from_last_cycle"]].to_dict("records"))
            .reset_index()
            .rename(columns={0: "Question Insight"})
        )

        final_json_str = json_df.to_json(orient="records")
        final_json = json.loads(final_json_str)
        return TransformationTrendSummary._clean_empty(final_json)

    # ----------------------------
    # Public quantiles (pandas 2 compatible)
    # ----------------------------
    def quantile_diff(self, train_data: pd.DataFrame, metric: str) -> pd.DataFrame:
        agg = (
            train_data.groupby(["Survey"], dropna=False)[metric]
            .agg(q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
            .reset_index()
        )
        return agg


if __name__ == "__main__":
    survey = 2
    train_data = pd.read_csv("insights/drivers/sample_data/transformation_trends_sample_data.csv", index_col=False)

    ob = TransformationTrendSummary(train_data=train_data, survey=survey)
    df_demo, prompt = ob.build_prompt()

    with open("insights/drivers/sample_output/transformation_trend_df_demo.json", "w") as f:
        json.dump(df_demo.to_dict(), f, indent=4)

    with open("insights/drivers/sample_output/transformation_trend_json_creation.json", "w") as f:
        json.dump(prompt, f, indent=4)

    convert_df = ob.convert_data(train_data.copy())
    with open("insights/drivers/sample_output/transformation_trend_convert_data.json", "w") as f:
        json.dump(convert_df.to_dict(), f, indent=4)

    diff = ob.quantile_diff(convert_df, "score_diff")
    print(diff)