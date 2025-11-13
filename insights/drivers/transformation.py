import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List


class TransformationSummary:
    """
    Pandas-expressive transformation summarizer for driver insights.

    Responsibilities:
    - Hold the raw train_data and selected survey id
    - Convert raw data into enriched, analysis-friendly DataFrame
    - Prepare an ordered subset tailored for JSON prompts
    - Emit clean JSON records (lists of dicts)
    """

    def __init__(self, train_data: pd.DataFrame, survey: int):
        self.train_data = train_data
        self.survey = survey

    def build_prompt(self) -> List[Dict[str, Any]]:
        """
        Execute the full pipeline end-to-end:
        1) Convert & enrich the input data
        2) Prepare the subset for the requested survey
        3) Create JSON records suitable for prompt consumption
        """
        df = self.convert_data(self.train_data.copy())
        df_demo_all = self._prepare_for_json(df, self.survey)
        return self._create_json(df_demo_all)

    @staticmethod
    def _clean_empty(d: Any) -> Any:
        """
        Recursively drop empty dict keys and list items from JSON-like structures.
        """
        if isinstance(d, dict):
            return {k: v for k, v in ((k, TransformationSummary._clean_empty(v)) for k, v in d.items()) if v}
        if isinstance(d, list):
            return [v for v in map(TransformationSummary._clean_empty, d) if v]
        return d

    # ------------------------------------------------------------------------------------
    # Data conversion (expressive, step-by-step; preserves original business logic)
    # ------------------------------------------------------------------------------------
    def convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the raw dataset with analysis fields using clear, pandas-friendly steps.

        The operations performed:
        - Enforce numeric types for score-related columns
        - Normalize the 'reversed' flag to boolean
        - Compute cycle-over-cycle score differences
        - Derive score zones using threshold percentiles (first row values)
        - Rank drivers per survey
        - Compute driver 'employee_confidence_level'
        - Compute score quantiles (q1, q3) per survey and flag outliers
        - Label question sentiment from flagged scores
        - Adjust sentiment for questions with reversed scales
        - Sanitize Description text
        - Drop helper columns
        """
        df = df.copy()

        # 1) Enforce numeric types where applicable
        numeric_cols = [
            "Score",
            "Survey",
            "Off Track Percentile",
            "On Track Percentile",
            "High Performance Percentile",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2) Normalize 'reversed' flag to boolean consistently
        if "reversed" in df.columns:
            df["reversed"] = (
                df["reversed"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"TRUE": True, "FALSE": False, "T": True, "F": False, "1": True, "0": False})
                .fillna(False)
            )

        # 3) Score difference within each (qcode, Type) group across cycles
        df["score_diff"] = df.groupby(["qcode", "Type"], dropna=False)["Score"].diff(1)

        # 4) Score zones computed from threshold columns (using first values)
        off_track = self._first_value_numeric(df["Off Track Percentile"]) if "Off Track Percentile" in df.columns else np.nan
        on_track = self._first_value_numeric(df["On Track Percentile"]) if "On Track Percentile" in df.columns else np.nan
        high_perf = self._first_value_numeric(df["High Performance Percentile"]) if "High Performance Percentile" in df.columns else np.nan

        # Start with the lowest zone, then progressively overwrite to higher zones
        df["score_zone"] = np.where(df["Score"] >= off_track, "Unsustainable Zone", "Off-Track Zone")
        df["score_zone"] = np.where(df["Score"] >= on_track, "On-Track Zone", df["score_zone"])
        df["score_zone"] = np.where(df["Score"] >= high_perf, "High Performance Zone", df["score_zone"])

        # 5) Rank drivers by Score per (Type, Survey), keep ranks only for Driver rows
        df["driver_rank"] = (
            df.groupby(["Type", "Survey"], dropna=False)["Score"]
            .rank(ascending=False, method="dense")
            .astype("Int64")
        )
        df.loc[df["Type"] != "Driver", "driver_rank"] = np.nan
        # Forward-fill within (Survey, driver) so Question rows inherit a driver's rank
        df["driver_rank"] = df.groupby(["Survey", "driver"], dropna=False)["driver_rank"].ffill()

        # 6) Remove duplicate rows to avoid noisy downstream merges
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        # 7) Employee confidence level buckets for Driver rows
        is_driver = df["Type"] == "Driver"
        df.loc[is_driver & (df["Score"] >= off_track), "employee_confidence_level"] = "Low"
        df.loc[is_driver & (df["Score"] >= on_track), "employee_confidence_level"] = "Moderate"
        df.loc[is_driver & (df["Score"] >= (high_perf - 10)), "employee_confidence_level"] = "Strong"
        df.loc[is_driver & (df["Score"] >= high_perf), "employee_confidence_level"] = "Very Strong"
        df.loc[is_driver & (df["Score"] < off_track), "employee_confidence_level"] = "Very Low"

        # 8) Compute survey-level quantiles (q1, q3) for Score and merge
        qscore = self._compute_quantiles(df, metric="Score")
        df = pd.merge(df, qscore, on=["Survey"], how="left")

        # 9) Flag outlier-like scores outside the interquartile range
        df["score_flag"] = np.where((df["Score"] < df["q1"]) | (df["Score"] > df["q3"]), df["Score"], np.nan)

        # 10) Label question sentiment from flagged Score vs thresholds
        is_question = df["Type"] == "Question"
        df.loc[is_question & (df["score_flag"] >= on_track), "Employee Perception"] = "Positive"
        df.loc[is_question & (df["score_flag"] >= high_perf), "Employee Perception"] = "Very Positive"
        df.loc[is_question & (df["score_flag"] < on_track), "Employee Perception"] = "Negative"
        df.loc[is_question & (df["score_flag"] < off_track), "Employee Perception"] = "Very Negative"

        # 11) Adjust sentiments for reversed-scale questions
        mask_reversed = df["reversed"].astype(bool) if "reversed" in df.columns else False
        pos_mask = df["Employee Perception"] == "Positive"
        vpos_mask = df["Employee Perception"] == "Very Positive"
        neg_mask = df["Employee Perception"] == "Negative"
        vneg_mask = df["Employee Perception"] == "Very Negative"

        df.loc[is_question & mask_reversed & pos_mask, "Employee Perception"] = "Negative"
        df.loc[is_question & mask_reversed & vpos_mask, "Employee Perception"] = "Very Negative"
        df.loc[is_question & mask_reversed & neg_mask, "Employee Perception"] = "Positive"
        df.loc[is_question & mask_reversed & vneg_mask, "Employee Perception"] = "Very Positive"

        # 12) Tidy Description text (strip bullets and leading spaces)
        if "Description" in df.columns:
            df["Description"] = df["Description"].str.replace("* ", "", case=False, regex=False)
            df["Description"] = df["Description"].str.replace("^ ", "", case=False, regex=False)

        # 13) Drop helper columns to keep the output DataFrame clean
        df = df.drop(columns=["q1", "q3"], errors="ignore")

        return df

    @staticmethod
    def _first_value_numeric(series: pd.Series) -> float:
        """
        Return the numeric value of the first element in a Series; NaN if empty or non-numeric.
        """
        arr = pd.to_numeric(series.values, errors="coerce")
        return float(arr[0]) if len(arr) else float("nan")

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
        - Drivers: label Status by rank, reconcile with score zones, then select top 2 and bottom 2 ranks
        - Questions: keep only rows with score_flag, label Status from Employee Perception
        - Lowercase Driver and Description for consistency; rename 'driver' -> 'Driver'
        - Return a combined, ordered DataFrame ready for JSON conversion
        """
        quest_col_names = ["Driver", "Type", "Description", "Employee Perception", "Status", "driver_rank"]
        driver_col_names = ["Driver", "Type", "employee_confidence_level", "Status", "driver_rank"]

        # Filter to the requested survey and normalize display columns
        df_demo = df[df["Survey"] == survey].reset_index(drop=True).drop_duplicates()
        if "driver" in df_demo.columns:
            df_demo["driver"] = df_demo["driver"].str.lower()
        if "Description" in df_demo.columns:
            df_demo["Description"] = df_demo["Description"].str.lower()
        df_demo = df_demo.rename(columns={"driver": "Driver"})

        # Driver Status from rank (dense ranks: 1 = best)
        is_driver = df_demo["Type"] == "Driver"
        rank_num = pd.to_numeric(df_demo["driver_rank"], errors="coerce")
        df_demo.loc[is_driver & (rank_num <= 3), "Status"] = "top performing"
        df_demo.loc[is_driver & (rank_num > 3), "Status"] = "low performing"

        # Reconcile driver Status with score zones
        in_good_zone = df_demo["score_zone"].isin(["High Performance Zone", "On-Track Zone"])
        in_bad_zone = df_demo["score_zone"].isin(["Unsustainable Zone", "Off-Track Zone"])
        df_demo.loc[in_good_zone & (df_demo["Status"] == "low performing"), "Status"] = "top performing"
        df_demo.loc[in_bad_zone & (df_demo["Status"] == "top performing"), "Status"] = "ahead of other drivers but still poor performance"

        # Question Status from sentiment
        is_question = df_demo["Type"] == "Question"
        positive = df_demo["Employee Perception"].isin(["Very Positive", "Positive"])
        negative = df_demo["Employee Perception"].isin(["Very Negative", "Negative"])
        df_demo.loc[is_question & positive, "Status"] = "top performing"
        df_demo.loc[is_question & negative, "Status"] = "low performing"

        # Questions: keep anomalies (score_flag present) and select fields
        df_demo_quest = df_demo[is_question].reset_index(drop=True)
        df_demo_quest = (
            df_demo_quest[(df_demo_quest["score_flag"].notnull()) & (df_demo_quest["Type"] != "Driver")][quest_col_names]
            .reset_index(drop=True)
        )

        # Drivers: select top 2 and bottom 2 by rank
        df_demo_driver = df_demo[is_driver].reset_index(drop=True)
        top_ranks = (
            df_demo_driver.sort_values("driver_rank", ascending=True)
            .head(2)["driver_rank"]
            .dropna()
            .unique()
            .tolist()
        )
        low_ranks = (
            df_demo_driver.sort_values("driver_rank", ascending=False)
            .head(2)["driver_rank"]
            .dropna()
            .unique()
            .tolist()
        )
        df_demo_driver = df_demo_driver[
            df_demo_driver["driver_rank"].isin(top_ranks) | df_demo_driver["driver_rank"].isin(low_ranks)
        ]
        df_demo_driver = df_demo_driver.sort_values(by=["Driver", "employee_confidence_level"])
        df_demo_driver = df_demo_driver.dropna(
            subset=["employee_confidence_level", "score_flag", "Employee Perception"], how="all"
        )
        df_demo_driver = df_demo_driver[driver_col_names]

        # Combine and order the final output
        df_demo_all = (
            pd.concat([df_demo_driver, df_demo_quest], ignore_index=True)
            .sort_values(["driver_rank", "Driver", "Status"])
            .reset_index(drop=True)
        )

        return df_demo_all

    @staticmethod
    def _create_json(df_demo_all: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert the prepared DataFrame to a list of JSON records and strip empty values.
        """
        final_json_str = df_demo_all.to_json(orient="records")
        final_json = json.loads(final_json_str)
        return TransformationSummary._clean_empty(final_json)

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
    train_data = pd.read_csv("insights/drivers/sample_data/transformation_sample_data.csv", index_col=False)

    ob = TransformationSummary(train_data=train_data, survey=survey)
    prompt = ob.build_prompt()
    convert_df = ob.convert_data(train_data.copy())

    with open("insights/drivers/sample_output/transformation_convert_data_v2.json", "w") as f:
        json.dump(convert_df.to_dict(), f, indent=4)

    json_creation = ob._create_json(ob._prepare_for_json(convert_df, survey))

    with open("insights/drivers/sample_output/transformation_json_creation_v2.json", "w") as f:
        json.dump(json_creation, f, indent=4)

    diff = ob.quantile_diff(convert_df, "Score")
    print(diff)