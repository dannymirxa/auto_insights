"""
Wrapper utilities for insights functions that convert pandas DataFrame returns
into JSON-serializable Python structures (lists/dicts) so callers receive JSON
friendly outputs instead of pandas objects.

This file wraps functions from either the package-style modules under
'src/insights' or the top-level 'insights' package depending on import
resolution used in the runtime environment.

All wrapper functions preserve the original function names and signatures
where sensible, but they convert DataFrame results into Python-native
structures (lists/dicts/floats/strings) that can be serialized with json.dumps.
"""
from typing import List, Dict, Any
import json
import pandas as pd

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try importing as a top-level 'insights' package (works if PYTHONPATH/project installed),
# otherwise fall back to the in-repo 'src.insights' package so scripts run from the repository root.
from src.insights import correlations, data, findings, hotspots

def preprocess_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(path, skiprows=1)
    df_qcode = pd.read_excel(path, header=None)
    df_qcode = data.preprocess_question_qcode(df_qcode)

    return df, df_qcode

# --- data.py wrappers ---
def data_availability(df, metric_columns: List[str]) -> Dict[str, Any]:
    """
    Wrapper around data.data_availability
    Returns the same dict the original function returns (JSON-serializable).
    """
    result = data.data_availability(df, metric_columns)
    # result is already a dict with lists/strings -> safe to return
    return result


# def preprocess_df(df, metric_columns: List[str], demographic_cols: List[str]) -> Dict[str, Any]:
#     """
#     Wrapper for data.preprocess_df.

#     Original returns a pandas DataFrame. This wrapper returns a JSON-serializable
#     list of records (one dict per row).
#     """
#     processed = data.preprocess_df(df, metric_columns=metric_columns, demographic_cols=demographic_cols)
#     return processed.to_dict(orient='records')


# def preprocess_question_qcode(df) -> Dict[str, Any]:
#     """
#     Wrapper for data.preprocess_question_qcode.

#     Returns list-of-dicts with keys ['question','qcode'].
#     """
#     processed = data.preprocess_question_qcode(df)
#     return processed.to_dict(orient='records')

def mean_score_by_metric_columns(df: pd.DataFrame, metric_columns: List[str]) -> Dict[str, Any]:
    df_metrics = df[metric_columns].mean().reset_index()
    
    return df_metrics.rename(columns={"index": "qcode", 0: "mean_score"}).to_dict(orient='records')


# --- hotspots.py wrappers ---
def hotspots_and_bright_spots_by_function_and_joblevel(df, metric_columns: List[str], num_spots: int = 3) -> Dict[str, Any]:
    """
    Wrapper: returns a dict with keys:
      - 'hotspots_by_function_and_joblevel'
      - 'bright_spots_by_function_and_joblevel'
    Each value is a list of records (dicts) JSON-ready.
    """
    result = hotspots.hotspots_and_bright_spots_by_function_and_joblevel(df, metric_columns, num_spots=num_spots)
    # results are already lists/dicts with native types; ensure numeric primitives are native
    return result


def hotspots_and_bright_spots_by_function(df, metric_columns: List[str], num_spots: int = 3) -> Dict[str, Any]:
    """
    Wrapper for hotspots_and_bright_spots_by_function
    """
    result = hotspots.hotspots_and_bright_spots_by_function(df, metric_columns, num_spots=num_spots)
    return result


def hotspots_and_bright_spots_by_joblevel(df, metric_columns: List[str], num_spots: int = 3) -> Dict[str, Any]:
    """
    Wrapper for hotspots_and_bright_spots_by_joblevel
    """
    result = hotspots.hotspots_and_bright_spots_by_joblevel(df, metric_columns, num_spots=num_spots)
    return result


def hotspots_and_bright_spots_by_driver(df, metric_columns: List[str], num_spots: int = 3) -> Dict[str, Any]:
    """
    Wrapper for hotspots_and_bright_spots_by_driver

    Returns:
      - 'hotspots_by_driver': list of records
      - 'bright_spots_by_driver': list of records
    """
    result = hotspots.hotspots_and_bright_spots_by_function_and_driver(df, metric_columns, num_spots=num_spots)
    return result


# --- correlations.py wrappers ---
def correlations_matrix(df, metric_columns: List[str]) -> Dict[str, Any]:
    """
    Wrapper for correlations.correlations_matrix.

    Original returns a pandas DataFrame (square). This wrapper returns a nested
    dict mapping row -> {col: value} with native floats or None.
    """
    corr = correlations.correlations_matrix(df, metric_columns)
    return corr.to_dict(orient='records')


def correlations_between_qcode_and_single_driver(df, df_questions_qcode, metric_columns: List[str]) -> Dict[str, Any]:
    """
    Wrapper for correlations.correlations_between_qcode_and_single_driver.

    Returns list-of-dicts with keys ['question', 'qcode', 'score'] ordered by score desc.
    """
    df_result = correlations.correlations_between_qcode_and_single_driver(df, df_questions_qcode, metric_columns)
    return df_result.to_dict(orient='records')

# -----------------------
# Example CLI/demo main
# -----------------------

def main():
    # Path to the Excel file (string, not a tuple)
    data_file = "data/TGPS Learning Activity Cycle.xlsx"
    # metric_columns should be a list/tuple of column names (no trailing commas)
    metric_columns = ["adb|info_managers", "adb|info_written", "ada|info_intranet"]
    demographic_cols = ['Job Level', 'Length of Service', 'Function', 'Sub-Function']

    df, df_qcode = preprocess_data(data_file)

    outputs = {}

    # 1) data_availability
    try:
        outputs["data_availability"] = data_availability(df, metric_columns)
    except Exception as e:
        outputs["data_availability_error"] = str(e)

    # 2) preprocess_df
    # try:
    #     outputs["preprocess_df"] = preprocess_df(df, metric_columns, demographic_cols)
    # except Exception as e:
    #     outputs["preprocess_df_error"] = str(e)

    # 3) preprocess_question_qcode
    # try:
    #     outputs["preprocess_question_qcode"] = preprocess_question_qcode(df_qcode)
    # except Exception as e:
    #     outputs["preprocess_question_qcode_error"] = str(e)

    # 4) hotspots wrappers
    try:
        outputs["hotspots_by_function_and_joblevel"] = hotspots_and_bright_spots_by_function_and_joblevel(df, metric_columns)
    except Exception as e:
        outputs["hotspots_by_function_and_joblevel_error"] = str(e)

    try:
        outputs["hotspots_by_function"] = hotspots_and_bright_spots_by_function(df, metric_columns)
    except Exception as e:
        outputs["hotspots_by_function_error"] = str(e)

    try:
        outputs["hotspots_by_joblevel"] = hotspots_and_bright_spots_by_joblevel(df, metric_columns)
    except Exception as e:
        outputs["hotspots_by_joblevel_error"] = str(e)

    try:
        outputs["hotspots_by_driver"] = hotspots_and_bright_spots_by_driver(df, metric_columns)
    except Exception as e:
        outputs["hotspots_by_driver_error"] = str(e)

    # 5) correlations wrappers
    try:
        outputs["correlations_matrix"] = correlations_matrix(df, metric_columns)
    except Exception as e:
        outputs["correlations_matrix_error"] = str(e)

    try:
        outputs["correlations_between_qcode_and_single_driver"] = correlations_between_qcode_and_single_driver(df, df_qcode, metric_columns)
    except Exception as e:
        outputs["correlations_between_qcode_and_single_driver_error"] = str(e)

        

    try:
        outputs["mean_score_by_metric_columns"] = mean_score_by_metric_columns(df, metric_columns)
    except Exception as e:
        outputs["mean_score_by_metric_columns_error"] = str(e)

    # Print a pretty JSON that is safe to serialize
    # print(json.dumps(outputs, indent=2, default=str))
    with open('ai_agent/utils/output_tools.json', 'w') as f:
        json.dump(outputs, f, indent=4)
    # df = correlations_between_qcode_and_single_driver(df, df_qcode, metric_columns)
    # print(df)

if __name__ == "__main__":
    main()

