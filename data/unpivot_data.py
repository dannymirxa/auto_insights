import pandas as pd

df = pd.read_csv('data/TIAA Cycle 3+4 (Aggregate)_modified.csv')
# df = pd.read_csv('data/TIAA Cycle 2 - Accenture_modified.csv')

qcodes = [
        "bpb|effectiveness",
        "bpb|experience",
        "bpb|customer_service",
        "bpb|change_cost_management",
        "sfa|benefits_revenue",
        "sfa|benefits_drive_savings",
        "sfa|benefits_lifetime_income",
        "sfa|benefits_innovative_prod",
        "sfa|career_opportunities",
        "tra|auto_issues_conflicting_priorities",
        "tra|issues_mgmt_support",
        "tra|ways_of_working",
        "adb|vision_lv1",
        "adb|vision_agree_lv1",
        "adb|conf_lv1_ldr",
        "enb|awareness",
        "tiaa|comment_vd",
        "adb|info_managers",
        "ada|info_colleagues",
        "adb|info_written",
        "ada|info_awareness_presentation",
        "enb|ldr_support_system",
        "enb|ldr_time_resources",
        "enb|conf_lv2_ldr",
        "sfb|current_change_mgmt",
        "rsb|quick_remedial",
        "tiaa|comment_bl",
        "llb|leads_implementation",
        "llb|performance_management",
        "llb|talents_utilised",
        "llb|conf_lv5_ldr",
        "llb|recognised_rewarded",
        "rsb|workgroup_systems",
        "rsb|workgroup_processes",
        "rsb|workgroup_capabilities",
        "rsb|workgroup_resources",
        "tiaa|comment_ss",
        "enb|committed_supportive",
        "eeb|teamwork",
        "tiaa|comment_tw",
        "llb|role_clarity",
        "llb|accountable",
        "llb|objectives_outcomes",
        "tiaa|comment_ac",
        "eeb|curiosity",
        "eeb|passion",
        "eeb|drive",
        "eeb|fear",
        "eeb|distress",
        "eeb|anger",
    ]

# qcodes = [
#     "bpb|experience",
#     "bpb|customer_service",
#     "bpb|change_cost_management",
#     "sfa|benefits_revenue",
#     "sfa|benefits_drive_savings",
#     "sfa|benefits_lifetime_income",
#     "sfa|benefits_innovative_prod",
#     "sfa|career_opportunities",
#     "tra|auto_issues_conflicting_priorities",
#     "tra|issues_mgmt_support",
#     "tra|ways_of_working",
#     "adb|vision_lv1",
#     "adb|vision_agree_lv1",
#     "adb|conf_lv1_ldr",
#     "enb|awareness",
#     "tiaa|comment_vd",
#     "adb|info_managers",
#     "ada|info_colleagues",
#     "adb|info_written",
#     "ada|info_awareness_presentation",
#     "enb|ldr_support_system",
#     "enb|ldr_time_resources",
#     "enb|conf_lv2_ldr",
#     "sfb|current_change_mgmt",
#     "rsb|quick_remedial",
#     "tiaa|comment_bl",
#     "llb|leads_implementation",
#     "llb|performance_management",
#     "llb|talents_utilised",
#     "llb|conf_lv5_ldr",
#     "llb|recognised_rewarded",
#     "rsb|workgroup_systems",
#     "rsb|workgroup_processes",
#     "rsb|workgroup_capabilities",
#     "rsb|workgroup_resources",
#     "tiaa|comment_ss",
#     "enb|committed_supportive",
#     "eeb|teamwork",
#     "tiaa|comment_tw",
#     "llb|role_clarity",
#     "llb|accountable",
#     "llb|objectives_outcomes",
#     "tiaa|comment_ac",
#     "eeb|curiosity",
#     "eeb|passion",
#     "eeb|drive",
#     "eeb|fear",
#     "eeb|distress",
#     "eeb|anger",
# ]

demographics_cols = ["People Manager", "Tenure Calc", "Work City", "Cycle", "Organization", "Department Name"]
# demographics_cols = ["Organization"]

def create_long_format(df: pd.DataFrame, qcodes: list, demographics_cols: list, output_path: str) -> pd.DataFrame:
    df[qcodes] = df[qcodes].apply(pd.to_numeric, errors="coerce")

    # 3) Compute per-respondent Score (mean across qcode columns)
    df["Score"] = df[qcodes].mean(axis=1)

    # 4) Create a single demographics dict column
    df["demographics"] = df[demographics_cols].to_dict(orient="records")

    # 5) Unpivot (melt) qcode columns to long format
    long = df.melt(
        id_vars=demographics_cols,
        value_vars=qcodes,
        var_name="qcode",
        value_name="score",
    )

    if "Cycle" in demographics_cols:
        long["Cycle"] = long["Cycle"].str.extract(r'(\d+)', expand=False)

    long.to_csv(output_path, index=False)
# 6) If you want demographics as a single column in the long output:
# long["demographics"] = long[demographics_cols].to_dict(orient="records")
# long = long[["demographics", "qcode", "value", "Score"]]

# # 7) Done. Example: inspect or write to CSV/Parquet
# print(long.head(10))
# long.to_parquet("data/tiaa_long.parquet")
# long.to_parquet("tiaa_long.parquet")
# long.to_csv("tiaa_long.csv", index=False)

create_long_format(df, qcodes, demographics_cols, "data/tiaa_3_4_long.csv")