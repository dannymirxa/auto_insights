import pandas as pd
from collections import defaultdict

def preprocess_question_qcode(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)

    # Take first two rows, transpose, and rename columns
    out = df.iloc[:2].T.copy()
    out.dropna(subset=[0], inplace=True)  # drop rows where qcode is NA
    out.columns = ['question', 'qcode']

    # Optional cleanups
    out = out.dropna(how='all').reset_index(drop=True)  # drop rows that are all NA
    out['question'] = out['question'].astype(str).str.strip()
    out['qcode'] = out['qcode'].astype(str).str.strip()


    return out

df = pd.read_excel("data/TGPS Learning Activity Cycle.xlsx", skiprows=1)

drivers = ['bpb|effectiveness', 'bpb|customer_service', 'bpb|change_cost_management', 'sfa|benefits_busarea_competitive', 'sfa|auto_benefits_system_efficiencies', 'sfa|benefits_customer_satis', 'sfa|auto_benefits_increased_sales', 'sfa|benefits_ways_of_working', 'sfb|lv1_success', 'tra|auto_issues_conflicting_priorities', 'tra|issues_mgmt_support', 'tra|issues_inability_adapt_change', 'tra|auto_issues_unclear_roles', 'tra|issues_technology', 'tra|issues_time_skills', 'tra|auto_issues_lack_of_resources', 'trb|internal_restructure', 'trb|new_way_of_working', 'trb|changing_size_shape', 'trb|overall_growth', 'trb|pace', 'adb|vision_lv1', 'adb|vision_agree_lv1', 'adb|conf_lv1_ldr', 'enb|awareness', 'adb|understand_purpose', 'adb|info_managers', 'adb|info_written', 'ada|info_intranet', 'enb|ldr_support_system', 'enb|ldr_time_resources', 'rsb|quick_remedial', 'sfb|current_change_mgmt', 'enb|conf_lv2_ldr', 'llb|leads_implementation', 'llb|performance_management', 'llb|talents_utilised', 'llb|recognised_rewarded', 'llb|conf_lv5_ldr', 'rsb|workgroup_systems', 'rsb|workgroup_processes', 'rsb|workgroup_capabilities', 'rsb|workgroup_resources', 'enb|lv4_implement', 'eeb|teamwork', 'llb|role_clarity', 'llb|accountable', 'llb|objectives_outcomes', 'eeb|passion', 'eeb|drive', 'eeb|fear', 'eeb|distress', 'eeb|anger']

vision_and_direction_drivers = ("adb|vision_lv1", "adb|vision_agree_lv1", "adb|conf_lv1_ldr", "enb|awareness", "adb|understand_purpose")
communication_drivers = ("enb|ldr_support_system", "enb|ldr_time_resources", "rsb|quick_remedial", "sfb|current_change_mgmt", "enb|conf_lv2_ldr")

def mean_score_by_respondent(df: pd.DataFrame, metric_columns: list) -> pd.DataFrame:
    # demographic_cols = ['respondent id', 'completed', 'completed_at', 'bmu', 'Fuction', 'Function', 'Job Level', 'Length of Service', 'Location', 'Sub-Function']
    df_mean_score = df.copy()
    df_mean_score['average_score'] = df_mean_score[list(metric_columns)].mean(axis=1)

    return df_mean_score


# print(mean_score_by_respondent(df, vision_and_direction_drivers)[['respondent id'] + list(vision_and_direction_drivers)+['average_score']].head())

# df = mean_score_by_respondent(df, vision_and_direction_drivers)


def correlations_between_qcode_and_driver(df: pd.DataFrame, metric_columns: list) -> pd.DataFrame:
    df_questions_qcode = preprocess_question_qcode("data/TGPS Learning Activity Cycle.xlsx")
    

    drivers = [column for column in df.columns if "|" in column and not "comments" in column]
    
    df_mean_score = df.copy()
    df_mean_score['average_score'] = df_mean_score[list(metric_columns)].mean(axis=1)

    df_score = defaultdict(float)
    for driver in list(set(drivers) - set(metric_columns)):
        df_score[driver] = float(round(df_mean_score[driver].corr(df_mean_score['average_score']), 2))

    # df_score_dict = dict(df_score)
    df_final = pd.DataFrame(list(dict(df_score).items()), columns=['qcode', 'score']).sort_values(by='score', ascending=False)

    df_final_questions = pd.DataFrame.merge(df_final, df_questions_qcode, on="qcode", how="inner")

    return df_final_questions[['question', 'qcode', 'score']]

print(correlations_between_qcode_and_driver(df, vision_and_direction_drivers))

# print([column for column in df.columns if not column.endswith('comments')])

# print(mean_score_by_respondent(df, communication_drivers)[['respondent id'] + list(communication_drivers)+['average_score']].head())