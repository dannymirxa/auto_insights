import pandas as pd
from typing import List, Dict

def data_availability(df: pd.DataFrame, metric_columns: List[str]) -> Dict:
    """
    Determine missing columns and non-numeric columns.
    Returns a dict with keys: missing_columns, numeric_available, non_numeric_or_empty
    """
    missing_columns = [c for c in metric_columns if c not in df.columns]
    numeric_available = []
    non_numeric_or_empty = []
    for c in metric_columns:
        if c not in df.columns:
            continue
        col_vals = pd.to_numeric(df[c], errors='coerce')
        if col_vals.dropna().shape[0] > 0:
            numeric_available.append(c)
        else:
            non_numeric_or_empty.append(c)
    return {
        "missing_columns": missing_columns,
        "numeric_available": numeric_available,
        "non_numeric_or_empty": non_numeric_or_empty,
    }


def preprocess_df(df: pd.DataFrame, metric_columns: List[str], demographic_cols: List[str]) -> pd.DataFrame:
    """
    Convert metrics to numeric and normalize demographic columns to strings with 'Unknown' for NaNs.
    This is the helper previously found in analysis_engine._preprocess_data.
    """
    for col in metric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in demographic_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    return df

def preprocess_question_qcode(df: pd.DataFrame) -> pd.DataFrame:

    # Take first two rows, transpose, and rename columns
    out = df.iloc[:2].T.copy()
    out.dropna(subset=[0], inplace=True)  # drop rows where qcode is NA
    out.columns = ['question', 'qcode']

    # Optional cleanups
    out = out.dropna(how='all').reset_index(drop=True)  # drop rows that are all NA
    out['question'] = out['question'].astype(str).str.strip()
    out['qcode'] = out['qcode'].astype(str).str.strip()

    return out

def df_qcode_agg_into_driver(df: pd.DataFrame, df_code: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Create qcode → driver mapping
    qcode_to_driver = df_code.set_index('qcode')['driver'].to_dict()

    # Step 2: Identify metric columns in df that are in df_code
    metric_columns = [col for col in df.columns if col in qcode_to_driver]

    # Step 3: Group metric columns by driver
    driver_groups: Dict[str, List[str]] = {}
    for qcode in metric_columns:
        driver = qcode_to_driver[qcode]
        driver_groups.setdefault(driver, []).append(qcode)

    # Step 4: Aggregate scores by driver (mean)
    driver_aggregates = pd.DataFrame()
    for driver, qcodes in driver_groups.items():
        driver_aggregates[driver] = df[qcodes].mean(axis=1)

    # Step 5: Combine with non-metric columns
    non_metric_columns = [col for col in df.columns if col not in metric_columns]
    df_final = pd.concat([df[non_metric_columns], driver_aggregates], axis=1)

    return df_final