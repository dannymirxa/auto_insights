import pandas as pd
import duckdb
from typing import List, Dict


def hotspots_and_bright_spots_by_function_and_joblevel(df: pd.DataFrame, metric_columns: List[str], num_spots: int = 3) -> Dict:
    """
    Identify hotspots (low performing) and bright spots (high performing) across Function & Job Level.
    Returns a dict with keys 'hotspots_by_function_and_joblevel' and 'bright_spots_by_function_and_joblevel'.
    """
    avg_expr = " + ".join([f'"{col}"' for col in metric_columns])
    avg_expr = f"({avg_expr}) / {len(metric_columns)}"

    query = f"""
        WITH scored AS (
            SELECT
                "Function",
                "Job Level",
                ROUND(AVG({avg_expr}), 2) AS average_score,
                COUNT(*) AS group_size
            FROM df
            GROUP BY "Function", "Job Level"
        )
        SELECT "Function", "Job Level", average_score
        FROM scored
        WHERE group_size >= 3
    """

    grouped = duckdb.query(query).to_df().set_index(['Function', 'Job Level'])['average_score']

    if grouped.empty:
        return {
            'hotspots_by_function_and_joblevel': [],
            'bright_spots_by_function_and_joblevel': []
        }

    hotspots = grouped.nsmallest(num_spots).reset_index().to_dict('records')
    bright_spots = grouped.nlargest(num_spots).reset_index().to_dict('records')
    return {'hotspots_by_function_and_joblevel': hotspots, 'bright_spots_by_function_and_joblevel': bright_spots}


def hotspots_and_bright_spots_by_function(df: pd.DataFrame, metric_columns: List[str], num_spots: int = 3) -> Dict:
    """
    Identify hotspots (low performing) and bright spots (high performing) by Function.
    Returns a dict with keys 'hotspots_by_function' and 'bright_spots_by_function'.
    """

    avg_expr = " + ".join([f'"{col}"' for col in metric_columns])
    avg_expr = f"({avg_expr}) / {len(metric_columns)}"
    
    query = f"""
        WITH scored AS (
            SELECT
                "Function",
                ROUND(AVG({avg_expr}), 2) AS average_score,
                COUNT(*) AS group_size
            FROM df
            GROUP BY "Function"
        )
        SELECT "Function", average_score
        FROM scored
        WHERE group_size >= 3
    """

    grouped = duckdb.query(query).to_df().set_index(['Function'])['average_score']

    if grouped.empty:
        return {'hotspots_by_function': [], 'bright_spots_by_function': []}
    
    hotspots = grouped.nsmallest(num_spots).reset_index().to_dict('records')
    bright_spots = grouped.nlargest(num_spots).reset_index().to_dict('records')
    return {'hotspots_by_function': hotspots, 'bright_spots_by_function': bright_spots}


def hotspots_and_bright_spots_by_joblevel(df: pd.DataFrame, metric_columns: List[str], num_spots: int = 3) -> Dict:
    """
    Identify hotspots (low performing) and bright spots (high performing) by Job Level.
    Returns a dict with keys 'hotspots_by_joblevel' and 'bright_spots_by_joblevel'.
    """
    avg_expr = " + ".join([f'"{col}"' for col in metric_columns])
    avg_expr = f"({avg_expr}) / {len(metric_columns)}"

    query = f"""
        WITH scored AS (
            SELECT
                "Job Level",
                ROUND(AVG({avg_expr}), 2) AS average_score,
                COUNT(*) AS group_size
            FROM df
            GROUP BY "Job Level"
        )
        SELECT "Job Level", average_score
        FROM scored
        WHERE group_size >= 3
    """

    grouped = duckdb.query(query).to_df().set_index(['Job Level'])['average_score']

    if grouped.empty:
        return {'hotspots_by_joblevel': [], 'bright_spots_by_joblevel': []}
    
    hotspots = grouped.nsmallest(num_spots).reset_index().to_dict('records')
    bright_spots = grouped.nlargest(num_spots).reset_index().to_dict('records')
    return {'hotspots_by_joblevel': hotspots, 'bright_spots_by_joblevel': bright_spots}


def hotspots_and_bright_spots_by_function_and_driver(df: pd.DataFrame, metric_columns: List[str], num_spots: int = 3) -> Dict:
    """
    For each metric (from metric_columns), compute mean scores grouped by 'Function' and
    return the lowest and highest performing Function groups per metric.

    Returns a dict with keys:
      - 'hotspots_by_driver': list of records {'Function','Metric','average_score'}
      - 'bright_spots_by_driver': list of records {'Function','Metric','average_score'}
    """
    df_scored = df.copy()

    hotspots_by_driver = []
    bright_spots_by_driver = []

    # Validate metric columns and compute per-metric group means by Function
    for metric in metric_columns:
        if metric not in df_scored.columns:
            continue

        query = f"""
                WITH scored AS (
                    SELECT
                        "Function",
                        CAST("{metric}" AS DOUBLE) AS val
                    FROM df
                ),
                grouped AS (
                    SELECT
                        "Function",
                        ROUND(AVG(val), 2) AS average_score,
                        COUNT(val) AS group_size
                    FROM scored
                    WHERE val IS NOT NULL
                    GROUP BY "Function"
                )
                SELECT "Function", average_score as "{metric}"
                FROM grouped
                WHERE group_size >= 3
            """

        grouped_mean = duckdb.query(query).to_df()

        if grouped_mean.empty:
            continue

        # smallest / largest for this metric
        smallest = grouped_mean.nsmallest(num_spots, columns=metric).reset_index()
        largest = grouped_mean.nlargest(num_spots, columns=metric).reset_index()

        for _, row in smallest.iterrows():
            hotspots_by_driver.append({
                'Function': row['Function'],
                'Metric': metric,
                'average_score': float(row[metric])
            })

        for _, row in largest.iterrows():
            bright_spots_by_driver.append({
                'Function': row['Function'],
                'Metric': metric,
                'average_score': float(row[metric])
            })

    return {
        'hotspots_by_driver': hotspots_by_driver,
        'bright_spots_by_driver': bright_spots_by_driver
    }