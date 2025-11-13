import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
import os
import sys

# Ensure project root on sys.path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from insights.drivers.transformation import TransformationSummary


class TurbulenceSummary(TransformationSummary):
    """
    Pandas-expressive turbulence summarizer.

    Responsibilities:
    - Inherit utilities from TransformationSummary (quantile_diff, _clean_empty)
    - Hold raw train_data and selected survey id
    - Convert raw data into an enriched DataFrame focused on turbulence-related drivers
    - Prepare a JSON structure summarizing changes, obstacles, and ranks
    """

    def __init__(self, train_data: pd.DataFrame, survey: int):
        super().__init__(train_data=train_data, survey=survey)
        self.train_data = train_data
        self.survey = survey

    def build_prompt(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Execute the full pipeline end-to-end:
        1) Convert & enrich the input data
        2) Prepare the subset for the requested survey
        3) Create JSON records suitable for prompt consumption

        Returns:
        (final_json, converted_df) to match prior expectations.
        """
        df = self.convert_data(self.train_data.copy())
        df_demo = self._prepare_for_json(df, self.survey)
        final_json = self._create_json(df_demo)
        return final_json, df

    @staticmethod
    def _remove_control_chars(s: Any) -> str:
        """
        Remove ASCII control characters (codepoints 0..31) from text.
        """
        txt = str(s)
        return "".join(ch for ch in txt if ord(ch) >= 32)

    # ------------------------------------------------------------------------------------
    # Data conversion (expressive, step-by-step; preserves original business logic)
    # ------------------------------------------------------------------------------------
    def convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the raw dataset with analysis fields using clear, pandas-friendly steps.

        Operations performed:
        - Coerce numeric types for relevant columns when present
        - Sort by Survey (if present) to stabilize diffs
        - Compute cycle-over-cycle score differences within each (Qcode/qcode, Type)
        - Rank drivers per (Type, Survey) with ascending rank (focus on turbulence/low scores)
        - Compute survey-level quantiles for score_diff and flag outlier-like changes (IQR OR abs change > 10)
        - Derive driver-level and question-level obstacle/perception labels for specific drivers
        - Support reversed scales for both Driver and Question rows
        - Label driver/question trends per driver category using score_diff or score_diff_flag
        - Sanitize Description text, replace explicit tokens, strip control characters
        - Drop helper quantile columns
        """
        df = df.copy()

        # 1) Coerce numeric columns when available
        numeric_cols = ["Score", "Survey", "Off Track Percentile", "On Track Percentile", "High Performance Percentile"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2) Sort by survey before diff (ensures temporal ordering)
        if "Survey" in df.columns:
            df = df.sort_values("Survey")

        # 3) Score diff per Qcode/Type (support legacy 'Qcode' vs 'qcode')
        group_cols = (
            ["Qcode", "Type"]
            if "Qcode" in df.columns
            else (["qcode", "Type"] if "qcode" in df.columns else ["Type"])
        )
        df["score_diff"] = df.groupby(group_cols, dropna=False)["Score"].diff(1)

        # 4) Driver rank: ascending (lower scores rank higher for turbulence focus)
        df["driver_rank"] = (
            df.groupby(["Type", "Survey"], dropna=False)["Score"]
            .rank(ascending=True, method="dense")
            .astype("Int64")
        )
        df.loc[df["Type"] != "Driver", "driver_rank"] = np.nan

        # 5) De-duplicate rows
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        # 6) Quantiles for score_diff (reuse parent named-agg) and outlier flag
        qdiff = self.quantile_diff(df, "score_diff")
        df = pd.merge(df, qdiff, on=["Survey"])
        df["score_diff_flag"] = np.where(
            (df["score_diff"] < df["q1"]) | (df["score_diff"] > df["q3"]) | (df["score_diff"].abs().gt(10)),
            df["score_diff"],
            np.nan,
        )

        # 7) Risks & Roadblocks mapping
        rr = "Risks & Roadblocks"

        # Driver-level perception (sequential overwrites preserve highest bucket)
        is_driver = df["Type"] == "Driver"
        rr_driver = df["Driver"] == rr
        df.loc[rr_driver & is_driver & (df["Score"] >= 25), "overall_employee_perception"] = "Some Obstacles"
        df.loc[rr_driver & is_driver & (df["Score"] >= 50), "overall_employee_perception"] = "Some Obstacles"
        df.loc[rr_driver & is_driver & (df["Score"] >= 75), "overall_employee_perception"] = "Major Obstacles"
        df.loc[rr_driver & is_driver & (df["Score"] < 25), "overall_employee_perception"] = "Low Obstacles"
        df.loc[rr_driver & is_driver & (df["Score"] < 10), "overall_employee_perception"] = "Low to No Obstacles"

        # Reversed scales for Drivers
        if "Reversed" in df.columns:
            rev_col = "Reversed"
        elif "reversed" in df.columns:
            rev_col = "reversed"
        else:
            rev_col = None

        if rev_col:
            rev_mask = df[rev_col].astype(bool)
            df.loc[rr_driver & rev_mask & is_driver & (df["Score"] >= 25), "overall_employee_perception"] = "Some Obstacles"
            df.loc[rr_driver & rev_mask & is_driver & (df["Score"] >= 50), "overall_employee_perception"] = "Some Obstacles"
            df.loc[rr_driver & rev_mask & is_driver & (df["Score"] >= 75), "overall_employee_perception"] = "Low Obstacles"
            df.loc[rr_driver & rev_mask & is_driver & (df["Score"] >= 90), "overall_employee_perception"] = "Low to No Obstacles"
            df.loc[rr_driver & rev_mask & is_driver & (df["Score"] < 25), "overall_employee_perception"] = "Major Obstacles"

        # Driver trends (reversed direction for Risks & Roadblocks)
        change_col_overall = "overall_change_from_last_cyle"  # typo preserved
        df.loc[rr_driver & is_driver & (df["score_diff"] > 3), change_col_overall] = "slightly decrease"
        df.loc[rr_driver & is_driver & (df["score_diff"] >= 10), change_col_overall] = "decrease"
        df.loc[rr_driver & is_driver & (df["score_diff"] >= 20), change_col_overall] = "significantly decrease"
        df.loc[rr_driver & is_driver & (df["score_diff"] < -3), change_col_overall] = "slightly increase"
        df.loc[rr_driver & is_driver & (df["score_diff"] < -10), change_col_overall] = "increase"
        df.loc[rr_driver & is_driver & (df["score_diff"] < -20), change_col_overall] = "significantly increase"

        # Question-level perception for Risks & Roadblocks
        is_question = df["Type"] == "Question"
        rr_question = df["Driver"] == rr
        df.loc[rr_question & is_question & (df["Score"] >= 25), "employee_perception"] = "Some Obstacles"
        df.loc[rr_question & is_question & (df["Score"] >= 50), "employee_perception"] = "Some Obstacles"
        df.loc[rr_question & is_question & (df["Score"] >= 75), "employee_perception"] = "Major Obstacles"
        df.loc[rr_question & is_question & (df["Score"] < 25), "employee_perception"] = "Low Obstacles"
        df.loc[rr_question & is_question & (df["Score"] < 10), "employee_perception"] = "Low to No Obstacles"

        # Reversed for Questions (Risks & Roadblocks)
        if rev_col:
            rev_mask = df[rev_col].astype(bool)
            df.loc[rr_question & rev_mask & is_question & (df["Score"] >= 25), "employee_perception"] = "Some Obstacles"
            df.loc[rr_question & rev_mask & is_question & (df["Score"] >= 50), "employee_perception"] = "Some Obstacles"
            df.loc[rr_question & rev_mask & is_question & (df["Score"] >= 75), "employee_perception"] = "Low Obstacles"
            df.loc[rr_question & rev_mask & is_question & (df["Score"] >= 90), "employee_perception"] = "Low to No Obstacles"
            df.loc[rr_question & rev_mask & is_question & (df["Score"] < 25), "employee_perception"] = "Major Obstacles"

        # Question trend for Risks & Roadblocks (use score_diff_flag thresholds)
        change_col_q = "change_from_last_cycle"
        df.loc[rr_question & is_question & (df["score_diff_flag"] > 3), change_col_q] = "slightly decrease"
        df.loc[rr_question & is_question & (df["score_diff_flag"] >= 10), change_col_q] = "decrease"
        df.loc[rr_question & is_question & (df["score_diff_flag"] >= 20), change_col_q] = "significantly decrease"
        df.loc[rr_question & is_question & (df["score_diff_flag"] < -3), change_col_q] = "slightly increase"
        df.loc[rr_question & is_question & (df["score_diff_flag"] < -10), change_col_q] = "increase"
        df.loc[rr_question & is_question & (df["score_diff_flag"] < -20), change_col_q] = "significantly increase"

        # 8) Amount of Change mapping
        aoc = "Amount of Change"

        # Driver perception
        aoc_driver = df["Driver"] == aoc
        df.loc[aoc_driver & is_driver & (df["Score"] >= 25), "overall_employee_perception"] = "Some Changes"
        df.loc[aoc_driver & is_driver & (df["Score"] >= 50), "overall_employee_perception"] = "Some Changes"
        df.loc[aoc_driver & is_driver & (df["Score"] >= 75), "overall_employee_perception"] = "Significant Changes"
        df.loc[aoc_driver & is_driver & (df["Score"] < 25), "overall_employee_perception"] = "Little Changes"

        # Reversed for Drivers
        if rev_col:
            rev_mask = df[rev_col].astype(bool)
            df.loc[aoc_driver & rev_mask & is_driver & (df["Score"] >= 25), "overall_employee_perception"] = "Some Changes"
            df.loc[aoc_driver & rev_mask & is_driver & (df["Score"] >= 50), "overall_employee_perception"] = "Some Changes"
            df.loc[aoc_driver & rev_mask & is_driver & (df["Score"] >= 75), "overall_employee_perception"] = "Little Changes"
            df.loc[aoc_driver & rev_mask & is_driver & (df["Score"] < 25), "overall_employee_perception"] = "Significant Changes"

        # Driver trends (Amount of Change)
        df.loc[aoc_driver & is_driver & (df["score_diff"] > 3), change_col_overall] = "slightly more"
        df.loc[aoc_driver & is_driver & (df["score_diff"] >= 10), change_col_overall] = "more"
        df.loc[aoc_driver & is_driver & (df["score_diff"] >= 20), change_col_overall] = "significantly more"
        df.loc[aoc_driver & is_driver & (df["score_diff"] < -3), change_col_overall] = "slightly lesser"
        df.loc[aoc_driver & is_driver & (df["score_diff"] < -10), change_col_overall] = "lesser"
        df.loc[aoc_driver & is_driver & (df["score_diff"] < -20), change_col_overall] = "significantly lesser"

        # Question perception (Amount of Change)
        aoc_question = df["Driver"] == aoc
        df.loc[aoc_question & is_question & (df["Score"] >= 25), "employee_perception"] = "Some Changes"
        df.loc[aoc_question & is_question & (df["Score"] >= 50), "employee_perception"] = "Some Changes"
        df.loc[aoc_question & is_question & (df["Score"] >= 75), "employee_perception"] = "Significant Changes"
        df.loc[aoc_question & is_question & (df["Score"] < 25), "employee_perception"] = "Little Changes"

        # Reversed for Questions (Amount of Change)
        if rev_col:
            rev_mask = df[rev_col].astype(bool)
            df.loc[aoc_question & rev_mask & is_question & (df["Score"] >= 25), "employee_perception"] = "Some Changes"
            df.loc[aoc_question & rev_mask & is_question & (df["Score"] >= 50), "employee_perception"] = "Some Changes"
            df.loc[aoc_question & rev_mask & is_question & (df["Score"] >= 75), "employee_perception"] = "Little Changes"
            df.loc[aoc_question & rev_mask & is_question & (df["Score"] < 25), "employee_perception"] = "Significant Changes"

        # Question trends (Amount of Change)
        df.loc[aoc_question & is_question & (df["score_diff_flag"] > 3), change_col_q] = "slightly more"
        df.loc[aoc_question & is_question & (df["score_diff_flag"] >= 10), change_col_q] = "more"
        df.loc[aoc_question & is_question & (df["score_diff_flag"] >= 20), change_col_q] = "significantly more"
        df.loc[aoc_question & is_question & (df["score_diff_flag"] < -3), change_col_q] = "slightly lesser"
        df.loc[aoc_question & is_question & (df["score_diff_flag"] < -10), change_col_q] = "lesser"
        df.loc[aoc_question & is_question & (df["score_diff_flag"] < -20), change_col_q] = "significantly lesser"

        # 9) Pace of Change mapping
        poc = "Pace of Change"

        # Question perception (Pace of Change)
        poc_question = df["Driver"] == poc
        df.loc[poc_question & is_question & (df["Score"] >= 25), "employee_perception"] = "Okay"
        df.loc[poc_question & is_question & (df["Score"] >= 50), "employee_perception"] = "Okay"
        df.loc[poc_question & is_question & (df["Score"] >= 75), "employee_perception"] = "Fast"
        df.loc[poc_question & is_question & (df["Score"] >= 90), "employee_perception"] = "Very Fast"
        df.loc[poc_question & is_question & (df["Score"] < 25), "employee_perception"] = "Too Slow"

        # Reversed for Questions (Pace of Change)
        if rev_col:
            rev_mask = df[rev_col].astype(bool)
            df.loc[poc_question & rev_mask & is_question & (df["Score"] >= 25), "employee_perception"] = "Okay"
            df.loc[poc_question & rev_mask & is_question & (df["Score"] >= 50), "employee_perception"] = "Okay"
            df.loc[poc_question & rev_mask & is_question & (df["Score"] >= 75), "employee_perception"] = "Too Slow"
            df.loc[poc_question & rev_mask & is_question & (df["Score"] < 25), "employee_perception"] = "Fast"
            df.loc[poc_question & rev_mask & is_question & (df["Score"] < 10), "employee_perception"] = "Very Fast"

        # Driver trends (Pace of Change)
        poc_driver = df["Driver"] == poc
        df.loc[poc_driver & is_driver & (df["score_diff"] > 3), change_col_overall] = "slightly faster"
        df.loc[poc_driver & is_driver & (df["score_diff"] >= 10), change_col_overall] = "faster"
        df.loc[poc_driver & is_driver & (df["score_diff"] >= 20), change_col_overall] = "significantly faster"
        df.loc[poc_driver & is_driver & (df["score_diff"] < -3), change_col_overall] = "slightly slower"
        df.loc[poc_driver & is_driver & (df["score_diff"] < -10), change_col_overall] = "slower"
        df.loc[poc_driver & is_driver & (df["score_diff"] < -20), change_col_overall] = "significantly slower"

        # Driver-level perception (Pace of Change)
        df.loc[poc_driver & is_driver & (df["Score"] >= 25), "overall_employee_perception"] = "Okay"
        df.loc[poc_driver & is_driver & (df["Score"] >= 50), "overall_employee_perception"] = "Okay"
        df.loc[poc_driver & is_driver & (df["Score"] >= 75), "overall_employee_perception"] = "Fast"
        df.loc[poc_driver & is_driver & (df["Score"] >= 90), "overall_employee_perception"] = "Very Fast"
        df.loc[poc_driver & is_driver & (df["Score"] < 25), "overall_employee_perception"] = "Too Slow"

        if rev_col:
            rev_mask = df[rev_col].astype(bool)
            df.loc[poc_driver & rev_mask & is_driver & (df["Score"] >= 25), "overall_employee_perception"] = "Okay"
            df.loc[poc_driver & rev_mask & is_driver & (df["Score"] >= 50), "overall_employee_perception"] = "Okay"
            df.loc[poc_driver & rev_mask & is_driver & (df["Score"] >= 75), "overall_employee_perception"] = "Too Slow"
            df.loc[poc_driver & rev_mask & is_driver & (df["Score"] < 25), "overall_employee_perception"] = "Fast"
            df.loc[poc_driver & rev_mask & is_driver & (df["Score"] < 10), "overall_employee_perception"] = "Very Fast"

        # Question trends (Pace of Change)
        df.loc[poc_question & is_question & (df["score_diff_flag"] > 3), change_col_q] = "slightly faster"
        df.loc[poc_question & is_question & (df["score_diff_flag"] >= 10), change_col_q] = "faster"
        df.loc[poc_question & is_question & (df["score_diff_flag"] >= 20), change_col_q] = "significantly faster"
        df.loc[poc_question & is_question & (df["score_diff_flag"] < -3), change_col_q] = "slightly slower"
        df.loc[poc_question & is_question & (df["score_diff_flag"] < -10), change_col_q] = "slower"
        df.loc[poc_question & is_question & (df["score_diff_flag"] < -20), change_col_q] = "significantly slower"

        # 10) Final cleanup
        if "Description" in df.columns:
            df["Description"] = df["Description"].str.replace("* ", "", case=False, regex=False)
            df["Description"] = df["Description"].str.replace("^ ", "", case=False, regex=False)
            df["Description"] = df["Description"].str.replace("lack of ", "", case=False, regex=False)
            df["Description"] = df["Description"].apply(TurbulenceSummary._remove_control_chars)

        df.replace(r"\bnan\b", np.nan, regex=True, inplace=True)
        df = df.drop(columns=["q1", "q3"], errors="ignore")

        return df

    # ------------------------------------------------------------------------------------
    # JSON preparation (structuring only; no business logic changes)
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _prepare_for_json(df: pd.DataFrame, survey: int) -> pd.DataFrame:
        """
        Create an ordered subset combining Driver and Question rows for the given survey.

        Rules applied:
        - Lowercase Driver and Description for consistency
        - Sort by driver_rank and Driver
        - Filter out Question rows lacking both perception and change signals
        - Forward-fill driver-level summaries (overall change and perception) and driver_rank within Driver groups
        - Keep only required output columns
        """
        col_names = [
            "Driver",
            "Type",
            "Description",
            "overall_employee_perception",
            "employee_perception",
            "overall_change_from_last_cyle",
            "change_from_last_cycle",
            "driver_rank",
        ]

        df_demo = df[df["Survey"] == survey].reset_index(drop=True).drop_duplicates()

        if "Driver" in df_demo.columns:
            df_demo["Driver"] = df_demo["Driver"].str.lower()
        if "Description" in df_demo.columns:
            df_demo["Description"] = df_demo["Description"].str.lower()

        # Sorting and filtering out rows with no question insight
        df_demo = df_demo.sort_values(by=["driver_rank", "Driver"])
        df_demo = df_demo[
            ~(
                df_demo["employee_perception"].isnull()
                & df_demo["change_from_last_cycle"].isnull()
                & (df_demo["Type"] == "Question")
            )
        ].reset_index(drop=True)

        # Forward-fill driver-level summaries
        df_demo["overall_change_from_last_cyle"] = df_demo.groupby(["Driver"])["overall_change_from_last_cyle"].ffill()
        df_demo["overall_employee_perception"] = df_demo.groupby(["Driver"])["overall_employee_perception"].ffill()
        df_demo["driver_rank"] = df_demo.groupby(["Driver"])["driver_rank"].ffill()

        # Retain only required columns present
        keep_cols = [c for c in col_names if c in df_demo.columns]
        df_demo = df_demo[keep_cols]

        return df_demo

    @staticmethod
    def _create_json(df_demo: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert the prepared DataFrame to grouped JSON with question-level insights.

        Grouping keys:
        - driver_rank
        - Driver
        - overall_change_from_last_cyle
        - overall_employee_perception

        Each group's value contains:
        - "Question Insight": a list of { Description, change_from_last_cycle, employee_perception }
        """
        json_df = (
            df_demo.groupby(
                ["driver_rank", "Driver", "overall_change_from_last_cyle", "overall_employee_perception"],
                dropna=False,
            )
            .apply(lambda x: x[["Description", "change_from_last_cycle", "employee_perception"]].to_dict("records"))
            .reset_index()
            .rename(columns={0: "Question Insight"})
        )

        final_json_str = json_df.to_json(orient="records")
        final_json = json.loads(final_json_str)
        # Use parent's cleanup utility
        return TransformationSummary._clean_empty(final_json)


if __name__ == "__main__":
    train_data = pd.read_csv("insights/drivers/sample_data/turbulence_sample_data.csv", index_col=False)
    survey = 2

    ts = TurbulenceSummary(train_data=train_data, survey=survey)
    js, convert_df = ts.build_prompt()

    with open("insights/drivers/sample_output/turbulence_summary_convert_data.json", "w") as f:
        json.dump(convert_df.to_dict(), f, indent=4)

    with open("insights/drivers/sample_output/turbulence_summary_js.json", "w") as f:
        json.dump(js, f, indent=4)