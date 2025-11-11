import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


class OutcomeSummary:
    """
    Stateful outcome summarizer. Holds train_data and survey to build prompt outputs.
    """

    def __init__(self, train_data: pd.DataFrame, survey: int):
        self.train_data = train_data
        self.survey = survey

    def build_prompt(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute full pipeline: convert data, prepare subset, and emit nested json.
        """
        df = self.convert_data(self.train_data.copy())
        df_demo = self._prepare_for_json(df, self.survey)
        quest_json = self._create_nested_json(df_demo)
        return df_demo, quest_json

    @staticmethod
    def _clean_empty(d):
        if isinstance(d, dict):
            return {k: v for k, v in ((k, OutcomeSummary._clean_empty(v)) for k, v in d.items()) if v}
        if isinstance(d, list):
            return [v for v in map(OutcomeSummary._clean_empty, d) if v]
        return d

    # ----------------------------
    # Data conversion (previously convert_data)
    # ----------------------------
    def convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates data transformations without changing underlying logic.
        """
        df = self._compute_score_diff(df)
        df = self._compute_score_zone(df)
        df = self._label_driver_change(df)

        # quantiles and flags for score_diff
        qdiff = self._compute_quantiles(df, metric="score_diff")
        df = df.merge(qdiff, on=["Survey"], how="left")
        df = self._flag_by_quantiles(df, metric="score_diff", flag_col="score_diff_flag")

        # label question trend from score_diff_flag
        df = self._label_question_trend(df)

        # clean strings
        df = self._clean_text_fields(df)

        # quantiles and flags for Score
        df = df.drop(columns=["q1", "q3"], errors="ignore")
        qscore = self._compute_quantiles(df, metric="Score")
        df = df.merge(qscore, on=["Survey"], how="left")
        df = self._flag_by_quantiles(df, metric="Score", flag_col="score_flag")

        # employee outlook / confidence levels based on percentiles
        df = self.convert_outlier_data(df)

        return df

    @staticmethod
    def _compute_score_diff(df: pd.DataFrame) -> pd.DataFrame:
        tr = df.groupby(["qcode", "Type"])["Score"].diff(1)
        df = df.copy()
        df.loc[:, "score_diff"] = tr.values
        return df

    @staticmethod
    def _first_value_numeric(series: pd.Series) -> float:
        # Preserve original logic: use first value as threshold across rows
        arr = pd.to_numeric(series.values, errors="coerce")
        return float(arr[0]) if len(arr) else float("nan")

    def _compute_score_zone(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        off_track = self._first_value_numeric(df["Off Track Percentile"])
        on_track = self._first_value_numeric(df["On Track percentile"])
        high_perf = self._first_value_numeric(df["High Performance percentile"])

        df.loc[:, "score_zone"] = np.where(df.Score >= off_track, "Unsustainable Zone", "")
        df.loc[:, "score_zone"] = np.where(df.Score >= on_track, "On-Track Zone", df.score_zone)
        df.loc[:, "score_zone"] = np.where(df.Score >= high_perf, "High Performance Zone", df.score_zone)
        return df

    @staticmethod
    def _label_driver_change(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        col = "change_from_last_cycle"
        df.loc[(df.Type == "Driver") & (df.score_diff > 0), col] = "slight improvement"
        df.loc[(df.Type == "Driver") & (df.score_diff >= 10), col] = "significant improvement"
        df.loc[(df.Type == "Driver") & (df.score_diff < 0), col] = "slight decline"
        df.loc[(df.Type == "Driver") & (df.score_diff < -10), col] = "significant decline"
        df.loc[(df.Type == "Driver") & (df.score_diff == 0), col] = "no change"
        return df

    @staticmethod
    def _compute_quantiles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        # pandas 2 compatible named aggregations
        agg = (
            df.groupby(["Survey"], dropna=False)[metric]
            .agg(q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
            .reset_index()
        )
        return agg

    @staticmethod
    def _flag_by_quantiles(df: pd.DataFrame, metric: str, flag_col: str) -> pd.DataFrame:
        df = df.copy()
        df.loc[:, flag_col] = np.where((df[metric] <= df["q1"]) | (df[metric] >= df["q3"]), df[metric], np.nan)
        return df

    @staticmethod
    def _label_question_trend(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        col = "employee_perception_trend"
        df.loc[(df.score_diff_flag >= 10) & (df.Type == "Question"), col] = "improvement"
        df.loc[(df.score_diff_flag >= 20) & (df.Type == "Question"), col] = "noticeable improvement"
        df.loc[(df.score_diff_flag >= 30) & (df.Type == "Question"), col] = "significant improvement"
        df.loc[(df.score_diff_flag < -10) & (df.Type == "Question"), col] = "decline"
        df.loc[(df.score_diff_flag < -20) & (df.Type == "Question"), col] = "noticeable decline"
        df.loc[(df.score_diff_flag < -30) & (df.Type == "Question"), col] = "significant decline"
        return df

    @staticmethod
    def _clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Question"] = df["Question"].str.replace(
            "How have the following changed in the last 6 months",
            "Changed in the last 6 months",
            case=False,
            regex=False,
        )
        df["Question"] = df["Question"].str.replace(
            "where do you see that most benefits will be delivered from the erp erp",
            "where do you see that most benefits will be delivered from the erp",
            case=False,
            regex=False,
        )
        df["Interrogation Topic"] = df["Interrogation Topic"].str.replace("Improved ", "", case=False, regex=False)
        df["Interrogation Topic"] = df["Interrogation Topic"].str.replace("Change in ", "", case=False, regex=False)
        df["Interrogation Topic"] = df["Interrogation Topic"].str.replace(
            "Your Function Areas effectiveness", "Function Areas effectiveness", case=False, regex=False
        )
        df["Interrogation Topic"] = df["Interrogation Topic"].str.replace(
            "The level of customer service internal or external your Team provides",
            "customer service",
            case=False,
            regex=False,
        )
        df["Interrogation Topic"] = df["Interrogation Topic"].str.replace(
            "Managing costs and resources in your Team", "Managing costs and resources", case=False, regex=False
        )
        return df

    # ----------------------------
    # JSON preparation (no data manipulation beyond structuring)
    # ----------------------------
    @staticmethod
    def _prepare_for_json(df: pd.DataFrame, survey: int) -> pd.DataFrame:
        df_demo = df[df.Survey == survey].reset_index(drop=True).drop_duplicates()
        df_demo = df_demo.rename(columns={"driver": "Performance Indicator"})

        # rank questions within indicator by score_flag (drivers rank 0)
        df_demo.loc[:, "question_rank"] = (
            df_demo.groupby(["Performance Indicator"])["score_flag"].rank(ascending=False)
        )
        df_demo.loc[df_demo.Type == "Driver", "question_rank"] = 0

        # sort and prune
        df_demo = df_demo.sort_values(by=["Performance Indicator", "current_confidence_level"])
        df_demo = df_demo.dropna(
            subset=["current_confidence_level", "employee_perception_trend", "score_flag", "Employee Outlook"],
            how="all",
        )

        # forward-fill to avoid repetition
        df_demo.loc[:, "current_confidence_level"] = df_demo["current_confidence_level"].ffill()
        df_demo.loc[:, "change_from_last_cycle"] = df_demo["change_from_last_cycle"].ffill()

        # build full question text
        df_demo.loc[df_demo.Type == "Question", "Full_Question"] = (
            df_demo["Question"] + " in terms of " + df_demo["Interrogation Topic"]
        )

        df_demo = df_demo.sort_values(["question_rank", "Performance Indicator"], ascending=True)
        df_demo = df_demo.reset_index(drop=True)

        # for JSON creation only
        df_demo = df_demo.fillna("")
        return df_demo

    @staticmethod
    def _create_nested_json(df_demo: pd.DataFrame) -> List[Dict[str, Any]]:
        json2 = (
            df_demo.groupby(
                ["Performance Indicator", "current_confidence_level", "change_from_last_cycle", "employee_perception_trend"],
                dropna=False,
            )
            .apply(lambda x: x[["Employee Outlook", "Full_Question"]].to_dict("records"))
            .reset_index()
            .rename(columns={0: "Details"})
        )

        quest_json_str = (
            json2.groupby(["Performance Indicator", "current_confidence_level", "change_from_last_cycle"])
            .apply(lambda x: x[["employee_perception_trend", "Details"]].to_dict("records"))
            .reset_index()
            .rename(columns={0: "Insight"})
            .to_json(orient="records")
        )

        quest_json = json.loads(quest_json_str)
        return OutcomeSummary._clean_empty(quest_json)

    # ----------------------------
    # Percentile-based outlook/confidence (logic unchanged)
    # ----------------------------
    def convert_outlier_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        on_track = self._first_value_numeric(df["On Track percentile"])
        high_perf = self._first_value_numeric(df["High Performance percentile"])
        off_track = self._first_value_numeric(df["Off Track Percentile"])

        # Employee Outlook (Questions)
        df.loc[(df.Type == "Question") & (df.Score >= on_track), "Employee Outlook"] = "Optimistic"
        df.loc[(df.Type == "Question") & (df.Score >= high_perf), "Employee Outlook"] = "Strongly optimistic"
        df.loc[(df.Type == "Question") & (df.Score < on_track), "Employee Outlook"] = "Low optimism"
        df.loc[(df.Type == "Question") & (df.Score < off_track), "Employee Outlook"] = "Not optimistic"

        # Confidence level (Drivers)
        df = df.drop(columns=["q1", "q3"], errors="ignore")
        df.loc[(df.Type == "Driver") & (df.Score >= off_track), "current_confidence_level"] = "Low"
        df.loc[(df.Type == "Driver") & (df.Score >= on_track), "current_confidence_level"] = "Moderate"
        df.loc[(df.Type == "Driver") & (df.Score >= (high_perf - 10)), "current_confidence_level"] = "Strong"
        df.loc[(df.Type == "Driver") & (df.Score >= high_perf), "current_confidence_level"] = "Very Strong"
        df.loc[(df.Type == "Driver") & (df.Score < off_track), "current_confidence_level"] = "Very Low"

        return df


if __name__ == "__main__":
    train_data = pd.read_csv("insights/drivers/output_sample_input_data.csv", index_col=False)
    ob = OutcomeSummary(train_data=train_data, survey=1)
    df_demo, js = ob.build_prompt()

    with open("insights/drivers/danial_outcome_summary_modified_df_demo.json", "w") as f:
        json.dump(df_demo.to_dict(), f, indent=4)

    with open("insights/drivers/danial_outcome_summary_modified_js.json", "w") as f:
        json.dump(js, f, indent=4)