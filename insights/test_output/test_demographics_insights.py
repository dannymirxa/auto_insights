import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
        
from insights.demographics import DemographicsInsights

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

    with open("insights/test_output/demogprahics/output_demographics.json", "w") as json_file:
        json.dump(output, json_file, indent=4)

    print(output)

if __name__=="__main__":
    main()