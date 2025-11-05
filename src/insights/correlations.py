import pandas as pd
from typing import List
from collections import defaultdict

def correlations_matrix(df: pd.DataFrame, metric_columns: List[str]) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations for provided metric_columns. Returns DataFrame.
    """
    return df[metric_columns].corr()

def correlations_between_qcode_and_single_driver(df: pd.DataFrame, df_questions_qcode: pd.DataFrame, metric_columns: list) -> pd.DataFrame:
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

def format_heatmap(corr_matrix: pd.DataFrame) -> str:
    """
    Produce a simple Markdown table with emojis representing correlation strength.
    """
    def get_emoji(val):
        if pd.isna(val): return "⚪"
        if val > 0.7: return "🟩"
        if val > 0.4: return "🟢"
        if val > 0.1: return "🟡"
        if val > -0.1: return "⚪"
        return "🔴"

    columns = [column.replace("|", "\|") for column in corr_matrix.columns.tolist()]
    header = "| Metric | " + " | ".join(columns) + " |"
    separator = "|-----|" + "-----|" * len(corr_matrix.columns)
    rows = [header, separator]
    for index, row in corr_matrix.iterrows():
        row_str = f"| **{index.replace("|", "\|")}** |" + " | ".join([f"{val:.2f} {get_emoji(val)}" for val in row]) + " |"
        rows.append(row_str)
    return "\n".join(rows)