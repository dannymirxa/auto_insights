import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


class TransformationTrendSummary:
    """
    Pandas-expressive trend transformation summarizer.

    Responsibilities:
    - Hold raw train_data and selected survey id
    - Convert raw data into an enriched, analysis-friendly DataFrame for trends
    - Prepare JSON records summarizing changes and ranks
    """

    def __init__(self, train_data: pd.DataFrame, survey: int):
        self.train_data = train_data
        self.survey = survey

    def build_prompt(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute the full pipeline end-to-end:
        1) Convert & enrich the input data
        2) Prepare the subset for the requested survey
        3) Create JSON records suitable for prompt consumption
        """
        df = self.convert_data(self.train_data.copy())
        df_demo = self._prepare_for_json(df, self.survey)
        final_json = self._create_json(df_demo)
        return df_demo, final_json

    @staticmethod
    def _clean_empty(d: Any) -> Any:
        """
        Recursively drop empty dict keys and list items from JSON-like structures.
        """
        if isinstance(d, dict):
            return {k: v for k, v in ((k, TransformationTrendSummary._clean_empty(v)) for k, v in d.items()) if v}
        if isinstance(d, list):
            return [v for v in map(TransformationTrendSummary._clean_empty, d) if v]
        return d

    # ------------------------------------------------------------------------------------
    # Data conversion (expressive, step-by-step; preserves original business logic)
    # ------------------------------------------------------------------------------------
    def convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the raw dataset with analysis fields using clear, pandas-friendly steps.

        Operations performed:
        - Sort by Survey (if present) to stabilize diffs
        - Compute cycle-over-cycle score differences within each (Qcode/qcode, Type)
        - Rank drivers per (Type, Survey)
        - Compute element-wise percentile thresholds and derive driver confidence levels
        - Compute survey-level quantiles for score_diff and flag outlier-like changes
        - Label question sentiment from Score compared against element-wise thresholds
        - Adjust sentiment for reversed scales (supports 'Reversed' and 'reversed')
        - Replace 'number' with 'level' for specific drivers in text fields
        - Label change-from-last-cycle for both Questions and Drivers using score_diff
        - Sanitize Description text, drop helper quantile columns
        """
        df = df.copy()

        # 1) Sort by Survey before diff to ensure correct temporal ordering
        if "Survey" in df.columns:
            df = df.sort_values("Survey")

        # 2) Score difference within each (Qcode/qcode, Type) group across cycles
        group_cols = ["Qcode", "Type"] if "Qcode" in df.columns else ["qcode", "Type"]
        df["score_diff"] = df.groupby(group_cols)["Score"].diff(1)

        # 3) Rank drivers by Score per (Type, Survey); keep ranks only for Driver rows
        df["driver_rank"] = (
            df.groupby(["Type", "Survey"])["Score"]
            .rank(ascending=False, method="dense")
            .astype("Int64")
        )
        df.loc[df["Type"] != "Driver", "driver_rank"] = np.nan

        # 4) Remove duplicate rows to avoid noisy downstream operations
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        # 5) Confidence level for drivers using element-wise thresholds (preserved logic)
        off_track_s = (
            pd.to_numeric(df["Off Track Percentile"], errors="coerce")
            if "Off Track Percentile" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        on_track_s = (
            pd.to_numeric(df["On Track Percentile"], errors="coerce")
            if "On Track Percentile" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        high_perf_s = (
            pd.to_numeric(df["High Performance Percentile"], errors="coerce")
            if "High Performance Percentile" in df.columns
            else pd.Series(np.nan, index=df.index)
        )

        is_driver = df["Type"] == "Driver"
        df.loc[is_driver & (df["Score"] >= off_track_s), "employee_confidence_level"] = "Low"
        df.loc[is_driver & (df["Score"] >= on_track_s), "employee_confidence_level"] = "Moderate"  # bucket 1
        df.loc[is_driver & (df["Score"] >= (high_perf_s - 10)), "employee_confidence_level"] = "Strong"  # bucket 2
        df.loc[is_driver & (df["Score"] >= high_perf_s), "employee_confidence_level"] = "Very Strong"
        df.loc[is_driver & (df["Score"] < off_track_s), "employee_confidence_level"] = "Very Low"

        # 6) Survey-level quantiles for score_diff and outlier flags (exclusive bounds)
        qdiff = self._compute_quantiles(df, metric="score_diff")
        df = pd.merge(df, qdiff, on=["Survey"])
        df["score_diff_flag"] = np.where((df["score_diff"] < df["q1"]) | (df["score_diff"] > df["q3"]), df["score_diff"], np.nan)

        # 7) Question sentiment from Score vs element-wise thresholds
        is_question = df["Type"] == "Question"
        df.loc[is_question & (df["Score"] >= on_track_s), "Employee Perception"] = "Positive"
        df.loc[is_question & (df["Score"] >= high_perf_s), "Employee Perception"] = "Very Positive"
        df.loc[is_question & (df["Score"] < on_track_s), "Employee Perception"] = "Negative"
        df.loc[is_question & (df["Score"] < off_track_s), "Employee Perception"] = "Very Negative"

        # 8) Adjust sentiments for reversed-scale questions (supports 'Reversed' and 'reversed')
        if "Reversed" in df.columns:
            rev = df["Reversed"].astype(bool)
        elif "reversed" in df.columns:
            rev = df["reversed"].astype(bool)
        else:
            rev = pd.Series(False, index=df.index)

        pos_mask = df["Employee Perception"] == "Positive"
        vpos_mask = df["Employee Perception"] == "Very Positive"
        neg_mask = df["Employee Perception"] == "Negative"
        vneg_mask = df["Employee Perception"] == "Very Negative"

        df.loc[is_question & rev & pos_mask, "Employee Perception"] = "Negative"
        df.loc[is_question & rev & vpos_mask, "Employee Perception"] = "Very Negative"
        df.loc[is_question & rev & neg_mask, "Employee Perception"] = "Positive"
        df.loc[is_question & rev & vneg_mask, "Employee Perception"] = "Very Positive"

        # 9) Replace 'number' -> 'level' for specific drivers in Question/Description
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

        # 10) Question score_diff trends (overwriting order preserves highest-priority label)
        change_col_q = "change_from_last_cycle"
        df.loc[is_question & (df["score_diff"] > 0), change_col_q] = "slightly better"
        df.loc[is_question & (df["score_diff"] >= 10), change_col_q] = "better"
        df.loc[is_question & (df["score_diff"] >= 20), change_col_q] = "significantly better"
        df.loc[is_question & (df["score_diff"] < 0), change_col_q] = "slightly worsened"
        df.loc[is_question & (df["score_diff"] < -10), change_col_q] = "worsened"
        df.loc[is_question & (df["score_diff"] < -20), change_col_q] = "significantly worsened"

        # 11) Driver score_diff trends (uses separate result column)
        change_col_d = "driver_performance_change_from_last_cycle"
        df.loc[is_driver & (df["score_diff"] > 0), change_col_d] = "slightly better"
        df.loc[is_driver & (df["score_diff"] >= 10), change_col_d] = "better"
        df.loc[is_driver & (df["score_diff"] >= 16), change_col_d] = "significantly better"
        df.loc[is_driver & (df["score_diff"] < 0), change_col_d] = "slightly worsened"
        df.loc[is_driver & (df["score_diff"] < -10), change_col_d] = "worsened"
        df.loc[is_driver & (df["score_diff"] < -20), change_col_d] = "significantly worsened"
        df.loc[is_driver & (df["score_diff"] == 0), change_col_d] = "no change"

        # 12) Tidy Description text (strip bullets and leading spaces)
        if "Description" in df.columns:
            df["Description"] = df["Description"].str.replace("* ", "", case=False, regex=False)
            df["Description"] = df["Description"].str.replace("^ ", "", case=False, regex=False)

        # 13) Drop helper quantile columns to keep the output DataFrame clean
        df = df.drop(columns=["q1", "q3"], errors="ignore")

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

    # ------------------------------------------------------------------------------------
    # JSON preparation (structuring only; no business logic changes)
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _prepare_for_json(df: pd.DataFrame, survey: int) -> pd.DataFrame:
        """
        Create an ordered subset combining Driver and Question rows for the given survey.

        Rules applied:
        - Lowercase Driver and Description for consistency
        - Filter to drivers with significant question movement (when score_diff has no nulls)
        - Sort by driver_rank and Driver, prune rows lacking all key fields
        - Rename driver_performance_change_from_last_cycle -> overall_change_from_last_cyle (typo preserved)
        - Forward-fill group-wise fields to avoid repetition
        - Keep only the required output columns
        """
        required_cols = [
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

        df_demo = df[df["Survey"] == survey].reset_index(drop=True).drop_duplicates()

        if "Driver" in df_demo.columns:
            df_demo["Driver"] = df_demo["Driver"].str.lower()
        if "Description" in df_demo.columns:
            df_demo["Description"] = df_demo["Description"].str.lower()

        # Filter relevant drivers: significant question movement (only if score_diff has no nulls)
        if df_demo["score_diff"].notnull().all():
            drivers_ls = (
                df_demo[
                    (df_demo["Type"] == "Question")
                    & ((df_demo["score_diff_flag"].notnull()) | (df_demo["score_diff"].abs().gt(10)))
                ]["Driver"]
                .unique()
                .tolist()
            )
            df_demo = df_demo[df_demo["Driver"].isin(drivers_ls)].reset_index(drop=True)

        # Sort and prune rows with no useful information across key fields
        df_demo = df_demo.sort_values(by=["driver_rank", "Driver"])
        df_demo = df_demo.dropna(
            subset=[
                "score_diff",
                "change_from_last_cycle",
                "Employee Perception",
                "driver_performance_change_from_last_cycle",
            ],
            how="all",
        )

        # Rename and forward-fill group-wise display fields
        df_demo = df_demo.rename(
            columns={"driver_performance_change_from_last_cycle": "overall_change_from_last_cyle"}
        )
        df_demo["overall_change_from_last_cyle"] = df_demo.groupby(["Driver"])["overall_change_from_last_cyle"].ffill()
        df_demo["employee_confidence_level"] = df_demo.groupby(["Driver"])["employee_confidence_level"].ffill()

        # Retain only required columns present in the data
        keep_cols = [c for c in required_cols if c in df_demo.columns]
        df_demo = df_demo[keep_cols]

        return df_demo

    @staticmethod
    def _create_json(df_demo: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert the prepared DataFrame to grouped JSON with question-level insights.

        Grouping keys (when present):
        - driver_rank
        - Driver
        - overall_change_from_last_cyle
        - employee_confidence_level

        Each group's value contains:
        - "Question Insight": a list of { Description, Employee Perception, change_from_last_cycle }
        """
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

    # ------------------------------------------------------------------------------------
    # Public quantiles (pandas 2 compatible)
    # ------------------------------------------------------------------------------------
    def quantile_diff(self, train_data: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Convenience wrapper to compute q1/q3 per Survey for any metric column.
        """
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