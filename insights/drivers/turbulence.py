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
    Stateful turbulence summarizer. Inherits utilities (quantile_diff, _clean_empty) from TransformationSummary.
    Holds train_data and survey; produces JSON prompt and processed data.
    """

    def __init__(self, train_data: pd.DataFrame, survey: int):
        super().__init__(train_data=train_data, survey=survey)
        self.train_data = train_data
        self.survey = survey

    def build_prompt(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Execute full pipeline: convert data, prepare subset, create JSON.
        Returns (final_json, converted_df) to match prior expectations.
        """
        df = self.convert_data(self.train_data.copy())
        df_demo = self._prepare_for_json(df, self.survey)
        final_json = self._create_json(df_demo)
        return final_json, df

    @staticmethod
    def _remove_control_chars(s: Any) -> str:
        txt = str(s)
        # remove ASCII control chars 0..31
        return "".join(ch for ch in txt if ord(ch) >= 32)

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

        # score diff per Qcode/Type (support legacy 'Qcode' vs 'qcode')
        group_cols = ["Qcode", "Type"] if "Qcode" in df.columns else (["qcode", "Type"] if "qcode" in df.columns else ["Type"])
        df.loc[:, "score_diff"] = df.groupby(group_cols)["Score"].diff(1)

        # driver rank: ascending (focus on turbulence/low scores)
        df.loc[:, "driver_rank"] = (
            df.groupby(["Type", "Survey"])["Score"].rank(ascending=True, method="dense").astype("Int64")
        )
        df.loc[df.Type != "Driver", "driver_rank"] = np.nan

        # de-duplicate
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        # quantiles for score_diff (reuse parent named-agg)
        qdiff = self.quantile_diff(df, "score_diff")
        df = pd.merge(df, qdiff, on=["Survey"])

        # outlier flag (IQR + absolute > 10)
        df.loc[:, "score_diff_flag"] = np.where(
            (df.score_diff < df.q1) | (df.score_diff > df.q3) | (df.score_diff.abs().gt(10)),
            df.score_diff,
            np.nan,
        )

        # Risks & Roadblocks
        driver_name = "Risks & Roadblocks"
        # Driver level perception
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 25),
            "overall_employee_perception",
        ] = "Some Obstacles"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 50),
            "overall_employee_perception",
        ] = "Some Obstacles"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 75),
            "overall_employee_perception",
        ] = "Major Obstacles"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score < 25),
            "overall_employee_perception",
        ] = "Low Obstacles"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score < 10),
            "overall_employee_perception",
        ] = "Low to No Obstacles"

        # Reversed for Drivers
        if "Reversed" in df.columns:
            rev_col = "Reversed"
        elif "reversed" in df.columns:
            rev_col = "reversed"
        else:
            rev_col = None

        if rev_col:
            rev_mask = df[rev_col] == True
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 25),
                "overall_employee_perception",
            ] = "Some Obstacles"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 50),
                "overall_employee_perception",
            ] = "Some Obstacles"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 75),
                "overall_employee_perception",
            ] = "Low Obstacles"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 90),
                "overall_employee_perception",
            ] = "Low to No Obstacles"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score < 25),
                "overall_employee_perception",
            ] = "Major Obstacles"

        # Driver trends (reversed direction for Risks & Roadblocks)
        df.loc[
            (df.score_diff > 3) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "slightly decrease"
        df.loc[
            (df.score_diff >= 10) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "decrease"
        df.loc[
            (df.score_diff >= 20) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "significantly decrease"
        df.loc[
            (df.score_diff < -3) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "slightly increase"
        df.loc[
            (df.score_diff < -10) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "increase"
        df.loc[
            (df.score_diff < -20) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "significantly increase"

        # Risks & Roadblocks questions
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 25),
            "employee_perception",
        ] = "Some Obstacles"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 50),
            "employee_perception",
        ] = "Some Obstacles"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 75),
            "employee_perception",
        ] = "Major Obstacles"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score < 25),
            "employee_perception",
        ] = "Low Obstacles"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score < 10),
            "employee_perception",
        ] = "Low to No Obstacles"

        # Reversed for Questions
        if rev_col:
            rev_mask = df[rev_col] == True
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 25),
                "employee_perception",
            ] = "Some Obstacles"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 50),
                "employee_perception",
            ] = "Some Obstacles"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 75),
                "employee_perception",
            ] = "Low Obstacles"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 90),
                "employee_perception",
            ] = "Low to No Obstacles"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score < 25),
                "employee_perception",
            ] = "Major Obstacles"

        # Question trend for Risks & Roadblocks (use score_diff_flag thresholds)
        df.loc[
            (df.score_diff_flag > 3) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "slightly decrease"
        df.loc[
            (df.score_diff_flag >= 10) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "decrease"
        df.loc[
            (df.score_diff_flag >= 20) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "significantly decrease"
        df.loc[
            (df.score_diff_flag < -3) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "slightly increase"
        df.loc[
            (df.score_diff_flag < -10) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "increase"
        df.loc[
            (df.score_diff_flag < -20) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "significantly increase"

        # Amount of Change
        driver_name = "Amount of Change"
        # Driver perception
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 25),
            "overall_employee_perception",
        ] = "Some Changes"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 50),
            "overall_employee_perception",
        ] = "Some Changes"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 75),
            "overall_employee_perception",
        ] = "Significant Changes"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score < 25),
            "overall_employee_perception",
        ] = "Little Changes"

        # Reversed for Drivers
        if rev_col:
            rev_mask = df[rev_col] == True
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 25),
                "overall_employee_perception",
            ] = "Some Changes"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 50),
                "overall_employee_perception",
            ] = "Some Changes"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 75),
                "overall_employee_perception",
            ] = "Little Changes"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score < 25),
                "overall_employee_perception",
            ] = "Significant Changes"

        # Driver trends
        df.loc[
            (df.score_diff > 3) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "slightly more"
        df.loc[
            (df.score_diff >= 10) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "more"
        df.loc[
            (df.score_diff >= 20) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "significantly more"
        df.loc[
            (df.score_diff < -3) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "slightly lesser"
        df.loc[
            (df.score_diff < -10) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "lesser"
        df.loc[
            (df.score_diff < -20) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "significantly lesser"

        # Amount of Change questions
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 25),
            "employee_perception",
        ] = "Some Changes"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 50),
            "employee_perception",
        ] = "Some Changes"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 75),
            "employee_perception",
        ] = "Significant Changes"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score < 25),
            "employee_perception",
        ] = "Little Changes"

        # Reversed for Questions
        if rev_col:
            rev_mask = df[rev_col] == True
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 25),
                "employee_perception",
            ] = "Some Changes"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 50),
                "employee_perception",
            ] = "Some Changes"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 75),
                "employee_perception",
            ] = "Little Changes"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score < 25),
                "employee_perception",
            ] = "Significant Changes"

        # Question trends (Amount of Change)
        df.loc[
            (df.score_diff_flag > 3) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "slightly more"
        df.loc[
            (df.score_diff_flag >= 10) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "more"
        df.loc[
            (df.score_diff_flag >= 20) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "significantly more"
        df.loc[
            (df.score_diff_flag < -3) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "slightly lesser"
        df.loc[
            (df.score_diff_flag < -10) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "lesser"
        df.loc[
            (df.score_diff_flag < -20) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "significantly lesser"

        # Pace of Change (Questions)
        driver_name = "Pace of Change"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 25),
            "employee_perception",
        ] = "Okay"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 50),
            "employee_perception",
        ] = "Okay"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 75),
            "employee_perception",
        ] = "Fast"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score >= 90),
            "employee_perception",
        ] = "Very Fast"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Question") & (df.Score < 25),
            "employee_perception",
        ] = "Too Slow"

        # Reversed for Questions (Pace of Change)
        if rev_col:
            rev_mask = df[rev_col] == True
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 25),
                "employee_perception",
            ] = "Okay"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 50),
                "employee_perception",
            ] = "Okay"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score >= 75),
                "employee_perception",
            ] = "Too Slow"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score < 25),
                "employee_perception",
            ] = "Fast"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Question") & (df.Score < 10),
                "employee_perception",
            ] = "Very Fast"

        # Driver trends (Pace of Change)
        df.loc[
            (df.score_diff > 3) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "slightly faster"
        df.loc[
            (df.score_diff >= 10) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "faster"
        df.loc[
            (df.score_diff >= 20) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "significantly faster"
        df.loc[
            (df.score_diff < -3) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "slightly slower"
        df.loc[
            (df.score_diff < -10) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "slower"
        df.loc[
            (df.score_diff < -20) & (df.Type == "Driver") & (df.Driver == driver_name),
            "overall_change_from_last_cyle",
        ] = "significantly slower"

        # Pace of Change (Driver-level perception)
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 25),
            "overall_employee_perception",
        ] = "Okay"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 50),
            "overall_employee_perception",
        ] = "Okay"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 75),
            "overall_employee_perception",
        ] = "Fast"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score >= 90),
            "overall_employee_perception",
        ] = "Very Fast"
        df.loc[
            (df.Driver == driver_name) & (df.Type == "Driver") & (df.Score < 25),
            "overall_employee_perception",
        ] = "Too Slow"

        if rev_col:
            rev_mask = df[rev_col] == True
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 25),
                "overall_employee_perception",
            ] = "Okay"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 50),
                "overall_employee_perception",
            ] = "Okay"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score >= 75),
                "overall_employee_perception",
            ] = "Too Slow"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score < 25),
                "overall_employee_perception",
            ] = "Fast"
            df.loc[
                (df.Driver == driver_name) & rev_mask & (df.Type == "Driver") & (df.Score < 10),
                "overall_employee_perception",
            ] = "Very Fast"

        # Question trends (Pace of Change)
        df.loc[
            (df.score_diff_flag > 3) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "slightly faster"
        df.loc[
            (df.score_diff_flag >= 10) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "faster"
        df.loc[
            (df.score_diff_flag >= 20) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "significantly faster"
        df.loc[
            (df.score_diff_flag < -3) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "slightly slower"
        df.loc[
            (df.score_diff_flag < -10) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "slower"
        df.loc[
            (df.score_diff_flag < -20) & (df.Type == "Question") & (df.Driver == driver_name),
            "change_from_last_cycle",
        ] = "significantly slower"

        # final cleanup
        if "Description" in df.columns:
            df["Description"] = df["Description"].str.replace("* ", "", case=False, regex=False)
            df["Description"] = df["Description"].str.replace("^ ", "", case=False, regex=False)
            df["Description"] = df["Description"].str.replace("lack of ", "", case=False, regex=False)
            df["Description"] = df["Description"].apply(TurbulenceSummary._remove_control_chars)

        df.replace(r"\bnan\b", np.nan, regex=True, inplace=True)
        df = df.drop(columns=["q1", "q3"], errors="ignore")

        return df

    # ----------------------------
    # JSON preparation (no data manipulation beyond structuring)
    # ----------------------------
    @staticmethod
    def _prepare_for_json(df: pd.DataFrame, survey: int) -> pd.DataFrame:
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

        df_demo = df[(df.Survey == survey)].reset_index(drop=True).drop_duplicates()

        if "Driver" in df_demo.columns:
            df_demo.loc[:, "Driver"] = df_demo["Driver"].str.lower()
        if "Description" in df_demo.columns:
            df_demo.loc[:, "Description"] = df_demo["Description"].str.lower()

        # sorting and filtering out rows with no question insight
        df_demo = df_demo.sort_values(by=["driver_rank", "Driver"])
        df_demo = df_demo[
            ~(
                (df_demo.employee_perception.isnull())
                & (df_demo.change_from_last_cycle.isnull())
                & (df_demo.Type == "Question")
            )
        ].reset_index(drop=True)

        # forward-fill driver-level summaries
        df_demo.loc[:, "overall_change_from_last_cyle"] = df_demo.groupby(["Driver"])[
            "overall_change_from_last_cyle"
        ].ffill()
        df_demo.loc[:, "overall_employee_perception"] = df_demo.groupby(["Driver"])[
            "overall_employee_perception"
        ].ffill()
        df_demo.loc[:, "driver_rank"] = df_demo.groupby(["Driver"])["driver_rank"].ffill()

        # retain only required columns present
        keep_cols = [c for c in col_names if c in df_demo.columns]
        df_demo = df_demo[keep_cols]

        return df_demo

    @staticmethod
    def _create_json(df_demo: pd.DataFrame) -> List[Dict[str, Any]]:
        json_df = (
            df_demo.groupby(
                ["driver_rank", "Driver", "overall_change_from_last_cyle", "overall_employee_perception"], dropna=False
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