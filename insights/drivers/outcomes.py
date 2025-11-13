import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


class OutcomeSummary:
    """
    Pandas-expressive outcome summarizer for driver/question trends.

    Responsibilities:
    - Hold raw train_data and selected survey id
    - Convert raw data into an enriched, analysis-friendly DataFrame
    - Prepare a nested JSON structure summarizing outcomes for prompts
    """

    def __init__(self, train_data: pd.DataFrame, survey: int):
        self.train_data = train_data
        self.survey = survey

    def build_prompt(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute the full pipeline end-to-end:
        1) Convert & enrich the input data
        2) Prepare the subset for the requested survey
        3) Create nested JSON records suitable for prompt consumption
        """
        df = self.convert_data(self.train_data.copy())
        df_demo = self._prepare_for_json(df, self.survey)
        quest_json = self._create_nested_json(df_demo)
        return df_demo, quest_json

    @staticmethod
    def _clean_empty(d: Any) -> Any:
        """
        Recursively drop empty dict keys and list items from JSON-like structures.
        """
        if isinstance(d, dict):
            return {k: v for k, v in ((k, OutcomeSummary._clean_empty(v)) for k, v in d.items()) if v}
        if isinstance(d, list):
            return [v for v in map(OutcomeSummary._clean_empty, d) if v]
        return d

    # ------------------------------------------------------------------------------------
    # Data conversion (expressive, step-by-step; preserves original business logic)
    # ------------------------------------------------------------------------------------
    def convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the raw dataset with analysis fields using clear, pandas-friendly steps.

        Operations performed:
        - Enforce numeric types for score-related columns
        - Compute cycle-over-cycle score differences
        - Derive score zones using percentile thresholds (first row values)
        - Label driver change descriptors from score_diff
        - Compute quantiles for score_diff and flag by IQR thresholds (inclusive at q1/q3)
        - Label question trend from score_diff_flag
        - Clean and normalize text fields for readability
        - Compute quantiles for Score and flag by IQR thresholds (inclusive at q1/q3)
        - Derive Employee Outlook (Questions) and current_confidence_level (Drivers) using percentile thresholds
        """
        df = df.copy()

        # 1) Enforce numeric types where applicable (keeps original column name variants)
        numeric_cols = [
            "Score",
            "Survey",
            "Off Track Percentile",
            "On Track percentile",
            "High Performance percentile",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2) Score difference within each (qcode, Type) group across cycles
        df = self._compute_score_diff(df)

        # 3) Score zones computed from threshold columns (using first values)
        df = self._compute_score_zone(df)

        # 4) Label driver change based on score_diff magnitude/direction
        df = self._label_driver_change(df)

        # 5) Quantiles and flags for score_diff (inclusive on bounds)
        qdiff = self._compute_quantiles(df, metric="score_diff")
        df = df.merge(qdiff, on=["Survey"], how="left")
        df = self._flag_by_quantiles(df, metric="score_diff", flag_col="score_diff_flag")

        # 6) Label question trend from flagged score_diff values
        df = self._label_question_trend(df)

        # 7) Clean text fields (Question and Interrogation Topic)
        df = self._clean_text_fields(df)

        # 8) Quantiles and flags for Score (inclusive on bounds)
        df = df.drop(columns=["q1", "q3"], errors="ignore")
        qscore = self._compute_quantiles(df, metric="Score")
        df = df.merge(qscore, on=["Survey"], how="left")
        df = self._flag_by_quantiles(df, metric="Score", flag_col="score_flag")

        # 9) Percentile-based outlook/confidence (logic unchanged)
        df = self.convert_outlier_data(df)

        return df

    @staticmethod
    def _compute_score_diff(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cycle-over-cycle difference in Score within each (qcode, Type) group.
        """
        df = df.copy()
        tr = df.groupby(["qcode", "Type"], dropna=False)["Score"].diff(1)
        df["score_diff"] = tr.values
        return df

    @staticmethod
    def _first_value_numeric(series: pd.Series) -> float:
        """
        Preserve original logic: use the first numeric value in the Series as a global threshold.
        """
        arr = pd.to_numeric(series.values, errors="coerce")
        return float(arr[0]) if len(arr) else float("nan")

    def _compute_score_zone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign score_zone using Off Track, On Track, and High Performance percentile thresholds.

        Notes:
        - Column names 'On Track percentile' and 'High Performance percentile' are preserved as-is (lowercase p).
        - Thresholds are taken from the first value of each column.
        """
        df = df.copy()

        off_track = self._first_value_numeric(df["Off Track Percentile"]) if "Off Track Percentile" in df.columns else np.nan
        on_track = self._first_value_numeric(df["On Track percentile"]) if "On Track percentile" in df.columns else np.nan
        high_perf = self._first_value_numeric(df["High Performance percentile"]) if "High Performance percentile" in df.columns else np.nan

        # Start with an empty zone, then progressively overwrite with higher zones
        df["score_zone"] = np.where(df["Score"] >= off_track, "Unsustainable Zone", "")
        df["score_zone"] = np.where(df["Score"] >= on_track, "On-Track Zone", df["score_zone"])
        df["score_zone"] = np.where(df["Score"] >= high_perf, "High Performance Zone", df["score_zone"])

        return df

    @staticmethod
    def _label_driver_change(df: pd.DataFrame) -> pd.DataFrame:
        """
        Label change_from_last_cycle for Driver rows based on score_diff magnitude/direction.
        """
        df = df.copy()
        col = "change_from_last_cycle"
        is_driver = df["Type"] == "Driver"

        # Improvements
        df.loc[is_driver & (df["score_diff"] > 0), col] = "slight improvement"
        df.loc[is_driver & (df["score_diff"] >= 10), col] = "significant improvement"

        # Declines
        df.loc[is_driver & (df["score_diff"] < 0), col] = "slight decline"
        df.loc[is_driver & (df["score_diff"] < -10), col] = "significant decline"

        # No change
        df.loc[is_driver & (df["score_diff"] == 0), col] = "no change"

        return df

    @staticmethod
    def _compute_quantiles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Compute q1 (25th percentile) and q3 (75th percentile) for the given metric per Survey.
        """
        agg = (
            df.groupby(["Survey"], dropna=False)[metric]
            .agg(q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
            .reset_index()
        )
        return agg

    @staticmethod
    def _flag_by_quantiles(df: pd.DataFrame, metric: str, flag_col: str) -> pd.DataFrame:
        """
        Flag values outside the interquartile range [q1, q3] (inclusive at the bounds).
        """
        df = df.copy()
        df[flag_col] = np.where((df[metric] <= df["q1"]) | (df[metric] >= df["q3"]), df[metric], np.nan)
        return df

    @staticmethod
    def _label_question_trend(df: pd.DataFrame) -> pd.DataFrame:
        """
        Label employee_perception_trend for Question rows using score_diff_flag magnitude.
        """
        df = df.copy()
        col = "employee_perception_trend"
        is_question = df["Type"] == "Question"

        # Improvements
        df.loc[is_question & (df["score_diff_flag"] >= 10), col] = "improvement"
        df.loc[is_question & (df["score_diff_flag"] >= 20), col] = "noticeable improvement"
        df.loc[is_question & (df["score_diff_flag"] >= 30), col] = "significant improvement"

        # Declines
        df.loc[is_question & (df["score_diff_flag"] < -10), col] = "decline"
        df.loc[is_question & (df["score_diff_flag"] < -20), col] = "noticeable decline"
        df.loc[is_question & (df["score_diff_flag"] < -30), col] = "significant decline"

        return df

    @staticmethod
    def _clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tidy common phrasing in 'Question' and 'Interrogation Topic' for clarity.
        """
        df = df.copy()

        if "Question" in df.columns:
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

        if "Interrogation Topic" in df.columns:
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

    # ------------------------------------------------------------------------------------
    # JSON preparation (structuring only; no business logic changes)
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _prepare_for_json(df: pd.DataFrame, survey: int) -> pd.DataFrame:
        """
        Create an ordered subset combining Driver and Question rows for the given survey.

        Rules applied:
        - Rename 'driver' -> 'Performance Indicator' for display consistency
        - Rank questions within each indicator by score_flag (drivers have rank 0)
        - Sort and prune by relevant fields; forward-fill levels to avoid repetition
        - Build Full_Question by concatenating Question with Interrogation Topic
        - Fill NAs for JSON-output convenience
        """
        df_demo = df[df["Survey"] == survey].reset_index(drop=True).drop_duplicates()
        df_demo = df_demo.rename(columns={"driver": "Performance Indicator"})

        # Rank questions within indicator by score_flag (drivers rank 0)
        df_demo["question_rank"] = (
            df_demo.groupby(["Performance Indicator"], dropna=False)["score_flag"].rank(ascending=False)
        )
        df_demo.loc[df_demo["Type"] == "Driver", "question_rank"] = 0

        # Sort and prune
        df_demo = df_demo.sort_values(by=["Performance Indicator", "current_confidence_level"])
        df_demo = df_demo.dropna(
            subset=["current_confidence_level", "employee_perception_trend", "score_flag", "Employee Outlook"],
            how="all",
        )

        # Forward-fill to avoid repetition across rows
        df_demo["current_confidence_level"] = df_demo["current_confidence_level"].ffill()
        df_demo["change_from_last_cycle"] = df_demo["change_from_last_cycle"].ffill()

        # Build full question text
        is_question = df_demo["Type"] == "Question"
        if {"Question", "Interrogation Topic"}.issubset(df_demo.columns):
            df_demo.loc[is_question, "Full_Question"] = (
                df_demo.loc[is_question, "Question"] + " in terms of " + df_demo.loc[is_question, "Interrogation Topic"]
            )

        df_demo = df_demo.sort_values(["question_rank", "Performance Indicator"], ascending=True)
        df_demo = df_demo.reset_index(drop=True)

        # For JSON creation only
        df_demo = df_demo.fillna("")
        return df_demo

    @staticmethod
    def _create_nested_json(df_demo: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Build nested JSON grouped by Performance Indicator, confidence level, and trend.

        Structure:
        [
          {
            "Performance Indicator": "...",
            "current_confidence_level": "...",
            "change_from_last_cycle": "...",
            "Insight": [
              {
                "employee_perception_trend": "...",
                "Details": [
                  { "Employee Outlook": "...", "Full_Question": "..." },
                  ...
                ]
              }
            ]
          },
          ...
        ]
        """
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
            json2.groupby(["Performance Indicator", "current_confidence_level", "change_from_last_cycle"], dropna=False)
            .apply(lambda x: x[["employee_perception_trend", "Details"]].to_dict("records"))
            .reset_index()
            .rename(columns={0: "Insight"})
            .to_json(orient="records")
        )

        quest_json = json.loads(quest_json_str)
        return OutcomeSummary._clean_empty(quest_json)

    # ------------------------------------------------------------------------------------
    # Percentile-based outlook/confidence (logic unchanged)
    # ------------------------------------------------------------------------------------
    def convert_outlier_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive Employee Outlook for Questions and current_confidence_level for Drivers using
        On Track / High Performance / Off Track percentile thresholds (first-value logic).
        """
        df = df.copy()

        on_track = self._first_value_numeric(df["On Track percentile"]) if "On Track percentile" in df.columns else np.nan
        high_perf = self._first_value_numeric(df["High Performance percentile"]) if "High Performance percentile" in df.columns else np.nan
        off_track = self._first_value_numeric(df["Off Track Percentile"]) if "Off Track Percentile" in df.columns else np.nan

        # Employee Outlook (Questions)
        is_question = df["Type"] == "Question"
        df.loc[is_question & (df["Score"] >= on_track), "Employee Outlook"] = "Optimistic"
        df.loc[is_question & (df["Score"] >= high_perf), "Employee Outlook"] = "Strongly optimistic"
        df.loc[is_question & (df["Score"] < on_track), "Employee Outlook"] = "Low optimism"
        df.loc[is_question & (df["Score"] < off_track), "Employee Outlook"] = "Not optimistic"

        # Confidence level (Drivers)
        df = df.drop(columns=["q1", "q3"], errors="ignore")
        is_driver = df["Type"] == "Driver"
        df.loc[is_driver & (df["Score"] >= off_track), "current_confidence_level"] = "Low"
        df.loc[is_driver & (df["Score"] >= on_track), "current_confidence_level"] = "Moderate"
        df.loc[is_driver & (df["Score"] >= (high_perf - 10)), "current_confidence_level"] = "Strong"
        df.loc[is_driver & (df["Score"] >= high_perf), "current_confidence_level"] = "Very Strong"
        df.loc[is_driver & (df["Score"] < off_track), "current_confidence_level"] = "Very Low"

        return df


if __name__ == "__main__":
    train_data = pd.read_csv("insights/drivers/output_sample_input_data.csv", index_col=False)
    ob = OutcomeSummary(train_data=train_data, survey=1)
    df_demo, js = ob.build_prompt()

    with open("insights/drivers/danial_outcome_summary_modified_df_demo.json", "w") as f:
        json.dump(df_demo.to_dict(), f, indent=4)

    with open("insights/drivers/danial_outcome_summary_modified_js.json", "w") as f:
        json.dump(js, f, indent=4)