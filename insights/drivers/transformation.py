import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List


class TransformationSummary:
    """
    Stateful transformation summarizer. Holds train_data and survey; produces prompt JSON.
    """

    def __init__(self, train_data: pd.DataFrame, survey: int):
        self.train_data = train_data
        self.survey = survey

    def build_prompt(self) -> List[Dict[str, Any]]:
        """
        Execute full pipeline: convert data, prepare subset, and emit JSON records.
        """
        df = self.convert_data(self.train_data.copy())
        df_demo_all = self._prepare_for_json(df, self.survey)
        return self._create_json(df_demo_all)

    @staticmethod
    def _clean_empty(d):
        if isinstance(d, dict):
            return {k: v for k, v in ((k, TransformationSummary._clean_empty(v)) for k, v in d.items()) if v}
        if isinstance(d, list):
            return [v for v in map(TransformationSummary._clean_empty, d) if v]
        return d

    # ----------------------------
    # Data conversion (previously convert_data)
    # ----------------------------
    def convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates data transformations without changing underlying logic.
        """
        df = df.copy()

        # enforce numeric types
        for _col in ["Score", "Survey", "Off Track Percentile", "On Track Percentile", "High Performance Percentile"]:
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors="coerce")

        # normalize reversed flag to boolean
        if "reversed" in df.columns:
            df["reversed"] = (
                df["reversed"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"TRUE": True, "FALSE": False, "T": True, "F": False, "1": True, "0": False})
                .fillna(False)
            )

        # cycle-over-cycle score diff
        df.loc[:, "score_diff"] = df.groupby(["qcode", "Type"])["Score"].diff(1)

        # score zones using first-value thresholds
        off_track = self._first_value_numeric(df["Off Track Percentile"])
        on_track = self._first_value_numeric(df["On Track Percentile"])
        high_perf = self._first_value_numeric(df["High Performance Percentile"])

        df.loc[:, "score_zone"] = np.where(df.Score >= off_track, "Unsustainable Zone", "Off-Track Zone")
        df.loc[:, "score_zone"] = np.where(df.Score >= on_track, "On-Track Zone", df.score_zone)
        df.loc[:, "score_zone"] = np.where(df.Score >= high_perf, "High Performance Zone", df.score_zone)

        # rank drivers by score per survey
        df.loc[:, "driver_rank"] = (
            df.groupby(["Type", "Survey"])["Score"].rank(ascending=False, method="dense").astype("Int64")
        )
        df.loc[df.Type != "Driver", "driver_rank"] = np.nan
        df.loc[:, "driver_rank"] = df.groupby(["Survey", "driver"])["driver_rank"].ffill()

        # deduplicate
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        # confidence level for drivers
        df.loc[(df.Type == "Driver") & (df.Score >= off_track), "employee_confidence_level"] = "Low"
        df.loc[(df.Type == "Driver") & (df.Score >= on_track), "employee_confidence_level"] = "Moderate"
        df.loc[(df.Type == "Driver") & (df.Score >= (high_perf - 10)), "employee_confidence_level"] = "Strong"
        df.loc[(df.Type == "Driver") & (df.Score >= high_perf), "employee_confidence_level"] = "Very Strong"
        df.loc[(df.Type == "Driver") & (df.Score < off_track), "employee_confidence_level"] = "Very Low"

        # quantiles for Score (pandas 2 named agg)
        qscore = self._compute_quantiles(df, metric="Score")
        df = pd.merge(df, qscore, on=["Survey"])

        # outlier flag outside [q1, q3]
        df.loc[:, "score_flag"] = np.where((df.Score < df.q1) | (df.Score > df.q3), df.Score, np.nan)

        # question sentiment based on flagged score vs thresholds
        df.loc[(df.Type == "Question") & (df.score_flag >= on_track), "Employee Perception"] = "Positive"
        df.loc[(df.Type == "Question") & (df.score_flag >= high_perf), "Employee Perception"] = "Very Positive"
        df.loc[(df.Type == "Question") & (df.score_flag < on_track), "Employee Perception"] = "Negative"
        df.loc[(df.Type == "Question") & (df.score_flag < off_track), "Employee Perception"] = "Very Negative"

        # adjust for reversed scales
        mask_question = df.Type == "Question"
        mask_reversed = df["reversed"].astype(bool) if "reversed" in df.columns else False
        mask_pos = df["Employee Perception"] == "Positive"
        mask_vpos = df["Employee Perception"] == "Very Positive"
        mask_neg = df["Employee Perception"] == "Negative"
        mask_vneg = df["Employee Perception"] == "Very Negative"

        df.loc[mask_question & mask_reversed & mask_pos, "Employee Perception"] = "Negative"
        df.loc[mask_question & mask_reversed & mask_vpos, "Employee Perception"] = "Very Negative"
        df.loc[mask_question & mask_reversed & mask_neg, "Employee Perception"] = "Positive"
        df.loc[mask_question & mask_reversed & mask_vneg, "Employee Perception"] = "Very Positive"

        # sanitize description
        if "Description" in df.columns:
            df["Description"] = df["Description"].str.replace("* ", "", case=False, regex=False)
            df["Description"] = df["Description"].str.replace("^ ", "", case=False, regex=False)

        # cleanup helper cols
        df = df.drop(columns=["q1", "q3"], errors="ignore")

        return df

    @staticmethod
    def _first_value_numeric(series: pd.Series) -> float:
        arr = pd.to_numeric(series.values, errors="coerce")
        return float(arr[0]) if len(arr) else float("nan")

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
        quest_col_names = ["Driver", "Type", "Description", "Employee Perception", "Status", "driver_rank"]
        driver_col_names = ["Driver", "Type", "employee_confidence_level", "Status", "driver_rank"]

        df_demo = df[(df.Survey == survey)].reset_index(drop=True).drop_duplicates()

        if "driver" in df_demo.columns:
            df_demo.loc[:, "driver"] = df_demo["driver"].str.lower()
        if "Description" in df_demo.columns:
            df_demo.loc[:, "Description"] = df_demo["Description"].str.lower()
        df_demo = df_demo.rename(columns={"driver": "Driver"})

        # driver status from rank
        df_demo.loc[
            (pd.to_numeric(df_demo["driver_rank"], errors="coerce") <= 3) & (df_demo.Type == "Driver"),
            "Status",
        ] = "top performing"
        df_demo.loc[
            (pd.to_numeric(df_demo["driver_rank"], errors="coerce") > 3) & (df_demo.Type == "Driver"),
            "Status",
        ] = "low performing"

        # reconcile with zones
        df_demo.loc[
            (df_demo.score_zone.isin(["High Performance Zone", "On-Track Zone"])) & (df_demo.Status == "low performing"),
            "Status",
        ] = "top performing"
        df_demo.loc[
            (df_demo.score_zone.isin(["Unsustainable Zone", "Off-Track Zone"])) & (df_demo.Status == "top performing"),
            "Status",
        ] = "ahead of other drivers but still poor performance"

        # question status from sentiment
        df_demo.loc[
            (df_demo["Employee Perception"].isin(["Very Positive", "Positive"])) & (df_demo.Type == "Question"),
            "Status",
        ] = "top performing"
        df_demo.loc[
            (df_demo["Employee Perception"].isin(["Very Negative", "Negative"])) & (df_demo.Type == "Question"),
            "Status",
        ] = "low performing"

        # questions: keep anomalies and select fields
        df_demo_quest = df_demo[df_demo.Type == "Question"].reset_index(drop=True)
        df_demo_quest = (
            df_demo_quest[(df_demo_quest.score_flag.notnull()) & (df_demo_quest.Type != "Driver")][quest_col_names]
            .reset_index(drop=True)
        )

        # drivers: select top2 and bottom2 ranks
        df_demo_driver = df_demo[df_demo.Type == "Driver"].reset_index(drop=True)
        top_driver_filter = (
            df_demo_driver.sort_values("driver_rank", ascending=True).head(2)["driver_rank"].unique().tolist()
        )
        low_driver_filter = (
            df_demo_driver.sort_values("driver_rank", ascending=False).head(2)["driver_rank"].unique().tolist()
        )
        df_demo_driver = df_demo_driver[
            (df_demo_driver["driver_rank"].isin(top_driver_filter))
            | (df_demo_driver["driver_rank"].isin(low_driver_filter))
        ]
        df_demo_driver = df_demo_driver.sort_values(by=["Driver", "employee_confidence_level"])
        df_demo_driver = df_demo_driver.dropna(
            subset=["employee_confidence_level", "score_flag", "Employee Perception"], how="all"
        )
        df_demo_driver = df_demo_driver[driver_col_names]

        # combine and order
        df_demo_all = (
            pd.concat([df_demo_driver, df_demo_quest], ignore_index=True)
            .sort_values(["driver_rank", "Driver", "Status"])
            .reset_index(drop=True)
        )

        return df_demo_all

    @staticmethod
    def _create_json(df_demo_all: pd.DataFrame) -> List[Dict[str, Any]]:
        final_json_str = df_demo_all.to_json(orient="records")
        final_json = json.loads(final_json_str)
        return TransformationSummary._clean_empty(final_json)

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
    train_data = pd.read_csv("insights/drivers/sample_data/transformation_sample_data.csv", index_col=False)

    ob = TransformationSummary(train_data=train_data, survey=survey)
    prompt = ob.build_prompt()
    convert_df = ob.convert_data(train_data.copy())

    with open("insights/drivers/sample_output/transformation_convert_data.json", "w") as f:
        json.dump(convert_df.to_dict(), f, indent=4)

    json_creation = ob._create_json(ob._prepare_for_json(convert_df, survey))

    with open("insights/drivers/sample_output/transformation_json_creation.json", "w") as f:
        json.dump(json_creation, f, indent=4)

    diff = ob.quantile_diff(convert_df, "Score")
    print(diff)