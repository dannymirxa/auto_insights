import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pandas as pd
import json
from collections import defaultdict
from typing_extensions import List
from insights.data_process import data

from insights.data_process import data

def preprocess_data(data_path: str, map_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(data_path, skiprows=1)
    df_map = pd.read_excel(map_path)
    df_qcode = pd.read_excel(data_path, header=None)
    df_qcode = data.preprocess_question_qcode(df_qcode)
    df_driver_qcode = pd.DataFrame.merge(df_map, df_qcode, on="qcode", how="inner")

    return df, df_driver_qcode

def get_columns_by_driver(drivers: List[str], config: List[dict]) -> List[str]:
    # Combine metric columns for all specified drivers.
    # Unknown drivers are ignored. Duplicates are removed while preserving order.
    index = {item["driver_name"]: item["metric_columns"] for item in config}

    combined: List[str] = []
    for driver in drivers:
        cols = index.get(driver)
        if not cols:
            continue
        for col in cols:
            if col not in combined:
                combined.append(col)
    return combined

RECOMMENDATION_MAP = {
    'default': (
        "Overall performance is stable. Continue monitoring key metrics and supporting leadership communication efforts."
    ),

    # Existing
    'leadership_trust': (
        "**Increase Leadership Visibility and Trust:** Confidence in leadership is a concern. "
        "Launch initiatives to build trust through transparency, such as hosting more authentic, unscripted Q&A sessions with the Executive Team "
        "and ensuring leaders are actively addressing employee feedback."
    ),
    'manager_support': (
        "**Strengthen Manager Support Systems:** Manager support appears to be a weak point. "
        "Implement a dedicated program to equip managers with the tools, talking points, and training necessary to lead their teams effectively through the transformation. "
        "This is a critical leverage point for success."
    ),
    'resource_allocation': (
        "**Review and Communicate Resource Allocation:** Concerns about the availability of time and resources are evident. "
        "Leadership should conduct a transparent review of project roadmaps and resource allocation to ensure teams are not overburdened. "
        "Communicate the outcomes of this review to alleviate concerns."
    ),
    'vision_clarity': (
        "**Enhance Vision Clarity and Purpose:** The understanding of the vision or its purpose is low. "
        "Shift communication from *what* is changing to *why* it's essential for the company's future. "
        "Use multiple channels and storytelling to connect the transformation to the company's core mission."
    ),
    'communication_effectiveness': (
        "**Overhaul a Key Communication Channel:** A primary communication channel (like the intranet or email) is underperforming. "
        "Charter a project to gather user feedback and revamp this channel to ensure it meets employee needs and serves as a reliable source of information."
    ),

    'team_leadership': (
        "**Develop Stronger Team Leadership Practices:** Gaps in implementation, performance management, and recognition are evident. "
        "Provide leadership training focused on coaching, talent utilization, and recognition programs to ensure team leaders are empowering and motivating their teams."
    ),
    'systems_processes': (
        "**Optimize Systems and Processes:** Employees report inefficiencies in workgroup systems and processes. "
        "Conduct a process-mapping exercise to identify bottlenecks and invest in system upgrades or workflow redesigns to improve productivity."
    ),
    'skills_staffing': (
        "**Address Skills and Staffing Gaps:** Concerns about capabilities and resources suggest teams may be under-resourced or lack critical skills. "
        "Launch targeted upskilling programs and review staffing allocations to ensure teams are equipped to deliver."
    ),
    'teamwork': (
        "**Reinforce Teamwork and Collaboration:** While some teams are supportive, there are signs of uneven collaboration. "
        "Introduce cross-functional projects, peer recognition programs, and team-building initiatives to strengthen cohesion."
    ),
    'accountability': (
        "**Clarify Roles and Accountability:** Ambiguity around role clarity, accountability, and objectives is undermining performance. "
        "Revisit role definitions, set measurable outcomes, and ensure accountability frameworks are consistently applied."
    ),
    'org_sentiment_positive': (
        "**Harness Positive Organizational Sentiment:** Curiosity, passion, and drive are strong cultural assets. "
        "Leverage these by creating innovation challenges, recognition programs, and opportunities for employees to contribute ideas to strategic initiatives."
    ),
    'org_sentiment_negative': (
        "**Address Negative Organizational Sentiment:** Fear, distress, and anger are emerging risks. "
        "Provide safe channels for employees to voice concerns, increase psychological safety, and ensure leaders acknowledge and address these emotions constructively."
    )
}

ANALYSIS_CONFIG = [
    {
        'driver_name': "Vision & Direction",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': [
            "adb|vision_lv1", "adb|vision_agree_lv1", "adb|conf_lv1_ldr", "enb|awareness"
        ],
        'metric_themes': {
            "adb|vision_lv1": "vision_clarity",
            "adb|vision_agree_lv1": "vision_clarity",
            "adb|conf_lv1_ldr": "leadership_trust",
            "enb|awareness": "vision_clarity"
        }
    },
    {
        'driver_name': "Communication",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': [
            "adb|info_managers", "ada|info_colleagues", "adb|info_written", "ada|info_awareness_presentation"
        ],
        'metric_themes': {
            "adb|info_managers": "manager_support",
            "ada|info_colleagues": "communication_effectiveness",
            "adb|info_written": "communication_effectiveness",
            "ada|info_awareness_presentation": "communication_effectiveness"
        }
    },
    {
        'driver_name': "Business Leadership",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': [
            "enb|ldr_support_system", "enb|ldr_time_resources", "enb|conf_lv2_ldr",
            "sfb|current_change_mgmt", "rsb|quick_remedial"
        ],
        'metric_themes': {
            "enb|ldr_support_system": "manager_support",
            "enb|ldr_time_resources": "resource_allocation",
            "enb|conf_lv2_ldr": "leadership_trust",
            "sfb|current_change_mgmt": "manager_support",
            "rsb|quick_remedial": "manager_support"
        }
    },
    {
        'driver_name': "Team Leadership",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': [
            "llb|leads_implementation", "llb|performance_management",
            "llb|talents_utilised", "llb|conf_lv5_ldr", "llb|recognised_rewarded"
        ],
        'metric_themes': {
            "llb|leads_implementation": "team_leadership",
            "llb|performance_management": "team_leadership",
            "llb|talents_utilised": "team_leadership",
            "llb|conf_lv5_ldr": "leadership_trust",
            "llb|recognised_rewarded": "team_leadership"
        }
    },
    {
        'driver_name': "Systems & Processes",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': ["rsb|workgroup_systems", "rsb|workgroup_processes"],
        'metric_themes': {
            "rsb|workgroup_systems": "systems_processes",
            "rsb|workgroup_processes": "systems_processes"
        }
    },
    {
        'driver_name': "Skills & Staffing",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': ["rsb|workgroup_capabilities", "rsb|workgroup_resources"],
        'metric_themes': {
            "rsb|workgroup_capabilities": "skills_staffing",
            "rsb|workgroup_resources": "skills_staffing"
        }
    },
    {
        'driver_name': "Teamwork",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': ["enb|committed_supportive", "eeb|teamwork"],
        'metric_themes': {
            "enb|committed_supportive": "teamwork",
            "eeb|teamwork": "teamwork"
        }
    },
    {
        'driver_name': "Accountability",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': ["llb|role_clarity", "llb|accountable", "llb|objectives_outcomes"],
        'metric_themes': {
            "llb|role_clarity": "accountability",
            "llb|accountable": "accountability",
            "llb|objectives_outcomes": "accountability"
        }
    },
    {
        'driver_name': "Organizational Sentiment",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': [
            "eeb|curiosity", "eeb|passion", "eeb|drive",
            "eeb|fear", "eeb|distress", "eeb|anger"
        ],
        'metric_themes': {
            "eeb|curiosity": "org_sentiment_positive",
            "eeb|passion": "org_sentiment_positive",
            "eeb|drive": "org_sentiment_positive",
            "eeb|fear": "org_sentiment_negative",
            "eeb|distress": "org_sentiment_negative",
            "eeb|anger": "org_sentiment_negative"
        }
    }
]

# from insights.demographics.hotspots import get_drivers_scores_agg_by_demographics, hotspots_and_bright_spots_drivers

import duckdb

drivers = ["Fear & Frustration", "Vision & Direction"]
        
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

def main():
    data_file = "/mnt/c/Projects/auto_insights/data/TIAA Cycle 3+4 (Aggregate).xlsx"
    map_file = "/mnt/c/Projects/auto_insights/data/TIAA Cycle 3+4 (Aggregate) driver qcode.xlsx"

    demographic_col = ["Location", "People Manager"]

    df, df_qcode = preprocess_data(data_file, map_file)
    df_driver = data.df_qcode_agg_into_driver(df, df_qcode)
    
    # df_final = get_drivers_scores_agg_by_demographics(df_driver, demographic_col, df_qcode['driver'].unique())

    # average_score = hotspots_and_bright_spots_drivers(df_driver, df_qcode['driver'].unique(), num_spots=3)

    insights = DemographicsInsights(df_driver, demographic_col, df_qcode['driver'].unique())
    output = insights.get_output()

    with open("insights/demographics/test/output.json", "w") as json_file:
        json.dump(output, json_file, indent=4)

    print(output)

if __name__=="__main__":
    main()