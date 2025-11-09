import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import json
from collections import defaultdict
from typing_extensions import List
from insights.data_process import data
        
from insights.driver import TransformationDriverInsights

def main():
    data_file = "/mnt/c/Projects/auto_insights/data/TIAA Cycle 3+4 (Aggregate).xlsx"
    map_file = "data/TGPS_driver_qcode_percentile_question.csv"

    demographic_col = ["Location", "People Manager"]


    df = pd.read_excel(data_file, skiprows=1)
    df_map = pd.read_csv(map_file)

    df_driver = data.df_qcode_agg_into_driver(df, df_map)
    
    insights = TransformationDriverInsights(df, demographic_col, df_map, num_spots=3)
    output = insights.get_output()

    with open("insights/test_output/drivers/output_driver.json", "w") as json_file:
        json.dump(output, json_file, indent=4)

    print(output)

if __name__=="__main__":
    main()