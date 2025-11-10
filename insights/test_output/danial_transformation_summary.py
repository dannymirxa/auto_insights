# TGPS Platform Viewpoints - Transformation Summary: line-by-line comments
import json  # Built-in library for working with JSON (serialization/deserialization)
import re    # Regular expressions; currently unused in this file
import numpy as np  # NumPy for vectorized numeric operations and conditions
import pandas as pd  # Pandas for DataFrame manipulation


class transformation_summary:
    """Encapsulates data conversion and JSON creation to prepare inputs for GenAI summarization."""

    def __init__(self):
        """Initializer; no state required for current operations."""
        pass  # No initialization logic needed

    def get_prompt(self, train_data: pd.DataFrame, survey):
        """Pipeline: convert raw data, build JSON payload for the given survey cycle."""
        
        train_data = self.convert_data(train_data)  # Normalize, rank, flag, and label dataset
        js = self.json_creation(train_data, survey)  # Build cleaned JSON records for GenAI

        return js  # Return list-of-dicts compatible with model input

    def convert_data(self, train_data: pd.DataFrame):
        """
        Convert numerical input data into format that can be processed by genAI models.
        Args:
            train_data (dataframe): Data that is required after post processing and the historical data of the surveys stored.

        Returns:
            train_data (dataframe): Processed data that genAI can use for insights generation.
        """
        # Coerce types for robust boolean and numeric operations
        train_data = train_data.copy()  # Work on a copy to avoid mutating caller's DataFrame
        # Enforce numeric dtypes in key columns for comparisons and ranking
        for _col in ['Score', 'Survey', 'Off Track Percentile', 'On Track Percentile', 'High Performance Percentile']:
            if _col in train_data.columns:  # Only convert if column exists
                train_data[_col] = pd.to_numeric(train_data[_col], errors='coerce')  # Non-numeric -> NaN
        # Normalize the 'reversed' flag to a strict boolean
        if 'reversed' in train_data.columns:
            train_data['reversed'] = (
                train_data['reversed']                 # Source values may be bool/str/int
                .astype(str)                           # Convert to string for uniform mapping
                .str.strip()                           # Remove leading/trailing spaces
                .str.upper()                           # Uppercase for case-insensitive mapping
                .map({'TRUE': True, 'FALSE': False, 'T': True, 'F': False, '1': True, '0': False})  # Map to booleans
                .fillna(False)                         # Default to False if unmapped
            )

        # Calculate cycle-over-cycle score difference per item (qcode x Type)
        train_data.loc[:, 'score_diff'] = train_data.groupby(['qcode', 'Type'])['Score'].diff(1)  # Current minus previous

        # Assign zone labels based on score thresholds (percentiles)
        train_data.loc[:, 'score_zone'] = np.where(
            train_data.Score >= pd.to_numeric(train_data['Off Track Percentile'].values, errors='coerce')[0],
            'Unsustainable Zone',
            'Off-Track Zone'
        )
        train_data.loc[:, 'score_zone'] = np.where(
            train_data.Score >= pd.to_numeric(train_data['On Track Percentile'].values, errors='coerce')[0],
            'On-Track Zone',
            train_data.score_zone
        )
        train_data.loc[:, 'score_zone'] = np.where(
            train_data.Score >= pd.to_numeric(train_data['High Performance Percentile'].values, errors='coerce')[0],
            'High Performance Zone',
            train_data.score_zone
        )

        # Rank drivers within each survey by score (dense ranking, highest score = rank 1)
        train_data.loc[:, 'driver_rank'] = train_data.groupby(['Type', 'Survey'])['Score'].rank(ascending=False, method='dense').astype('Int64')
        train_data.loc[train_data.Type != 'Driver', 'driver_rank'] = np.NaN  # Only drivers have ranks
        train_data.loc[:, 'driver_rank'] = train_data.groupby(['Survey', 'driver'])['driver_rank'].ffill()  # Forward-fill within driver group

        # Remove duplicate rows after transformation to stabilize downstream steps
        new_train_data = train_data.drop_duplicates(keep='first').reset_index(drop=True)

        # Map driver Score to employee_confidence_level buckets based on thresholds
        new_train_data.loc[
            (new_train_data.Type == 'Driver') &
            (new_train_data.Score >= pd.to_numeric(new_train_data['Off Track Percentile'].values, errors='coerce')[0]),
            'employee_confidence_level'
        ] = 'Low'
        new_train_data.loc[
            (new_train_data.Type == 'Driver') &
            (new_train_data.Score >= pd.to_numeric(new_train_data['On Track Percentile'].values, errors='coerce')[0]),
            'employee_confidence_level'
        ] = 'Moderate'  # Bucket 1: On Track
        new_train_data.loc[
            (new_train_data.Type == 'Driver') &
            (new_train_data.Score >= pd.to_numeric(new_train_data['High Performance Percentile'].values, errors='coerce')[0] - 10),
            'employee_confidence_level'
        ] = 'Strong'    # Bucket 2: Near High Performance
        new_train_data.loc[
            (new_train_data.Type == 'Driver') &
            (new_train_data.Score >= pd.to_numeric(new_train_data['High Performance Percentile'].values, errors='coerce')[0]),
            'employee_confidence_level'
        ] = 'Very Strong'  # Bucket 3: High Performance
        new_train_data.loc[
            (new_train_data.Type == 'Driver') &
            (new_train_data.Score < pd.to_numeric(new_train_data['Off Track Percentile'].values, errors='coerce')[0]),
            'employee_confidence_level'
        ] = 'Very Low'  # Below Off-Track threshold

        # Compute question-level anomaly flags via interquartile range (q1/q3)
        quantile_df = self.quantile_diff(new_train_data, "Score")  # Per-survey Q1 and Q3
        quantile_df.columns = quantile_df.columns.droplevel(0)     # Flatten multiindex columns
        quantile_df = quantile_df.reset_index()                    # Materialize 'Survey' as column
        new_train_data = pd.merge(new_train_data, quantile_df, on=['Survey'])  # Attach q1/q3 to each row

        # Flag outliers: keep Score when outside [q1, q3], else NaN (no flag)
        new_train_data.loc[:, 'score_flag'] = np.where(
            (new_train_data.Score < new_train_data.q1) | (new_train_data.Score > new_train_data.q3),
            new_train_data.Score,
            np.NaN
        )

        # Label question sentiment based on flagged score vs percentiles
        new_train_data.loc[
            (new_train_data.Type == 'Question') &
            (new_train_data.score_flag >= pd.to_numeric(new_train_data['On Track Percentile'].values, errors='coerce')[0]),
            'Employee Perception'
        ] = "Positive"
        new_train_data.loc[
            (new_train_data.Type == 'Question') &
            (new_train_data.score_flag >= pd.to_numeric(new_train_data['High Performance Percentile'].values, errors='coerce')[0]),
            'Employee Perception'
        ] = "Very Positive"
        new_train_data.loc[
            (new_train_data.Type == 'Question') &
            (new_train_data.score_flag < pd.to_numeric(new_train_data['On Track Percentile'].values, errors='coerce')[0]),
            'Employee Perception'
        ] = "Negative"
        new_train_data.loc[
            (new_train_data.Type == 'Question') &
            (new_train_data.score_flag < pd.to_numeric(new_train_data['Off Track Percentile'].values, errors='coerce')[0]),
            'Employee Perception'
        ] = "Very Negative"

        # Adjust sentiment for reversed scales (e.g., higher score means worse sentiment)
        mask_question = new_train_data.Type == 'Question'            # Only for questions
        mask_reversed = new_train_data['reversed'].astype(bool)      # Items with reversed scale
        mask_pos = new_train_data['Employee Perception'] == 'Positive'
        mask_vpos = new_train_data['Employee Perception'] == 'Very Positive'
        mask_neg = new_train_data['Employee Perception'] == 'Negative'
        mask_vneg = new_train_data['Employee Perception'] == 'Very Negative'

        new_train_data.loc[mask_question & mask_reversed & mask_pos, 'Employee Perception'] = "Negative"        # Flip Positive -> Negative
        new_train_data.loc[mask_question & mask_reversed & mask_vpos, 'Employee Perception'] = "Very Negative"  # Flip Very Positive -> Very Negative
        new_train_data.loc[mask_question & mask_reversed & mask_neg, 'Employee Perception'] = "Positive"        # Flip Negative -> Positive
        new_train_data.loc[mask_question & mask_reversed & mask_vneg, 'Employee Perception'] = "Very Positive"  # Flip Very Negative -> Very Positive

        # Sanitize free-text Description by removing leading markers commonly found in bullet lists
        new_train_data['Description'] = new_train_data['Description'].str.replace("* ", "", case=False)  # Remove asterisk bullets
        new_train_data['Description'] = new_train_data['Description'].str.replace("^ ", "", case=False)  # Remove caret bullets

        new_train_data.drop(columns=['q1', 'q3'], inplace=True)  # Drop helper quantile columns after use

        return new_train_data  # Return fully prepared dataset
    
    def json_creation(self, new_train_data, survey):
        """
        Further clean up of processed data and returns a json in string format to be fed to the genAI model.
        Args:
            train_data (dataframe): Data that is required after post processing and the historical data of the surveys stored.
            survey (int): Integer indicating current survey cycle.

        Returns:
            final_json (str): json input in string format to be fed to GenAI model.
        """
        
        quest_col_names = ['Driver', 'Type', 'Description', 'Employee Perception', 'Status', 'driver_rank']  # Output fields for questions
        driver_col_names = ['Driver', 'Type', 'employee_confidence_level', 'Status', 'driver_rank']          # Output fields for drivers

        # Filter to the requested survey cycle and deduplicate
        df_demo = new_train_data[(new_train_data.Survey == survey)].reset_index(drop=True).drop_duplicates()
        df_demo.loc[:, 'driver'] = df_demo.driver.str.lower()        # Normalize driver names to lowercase for consistency
        df_demo.loc[:, 'Description'] = df_demo.Description.str.lower()  # Normalize descriptions to lowercase
        df_demo.rename(columns={'driver':'Driver'}, inplace=True)     # Promote 'driver' column to canonical 'Driver'

        # Derive binary performance Status for drivers based on rank (top 3 vs others)
        df_demo.loc[(pd.to_numeric(df_demo.driver_rank, errors='coerce') <= 3) & (df_demo.Type == 'Driver'), 'Status'] = 'top performing'
        df_demo.loc[(pd.to_numeric(df_demo.driver_rank, errors='coerce') > 3) & (df_demo.Type == 'Driver'), 'Status'] = 'low performing'

        # Correct Status using zone context: reconcile high-score zones with low-performing statuses and vice versa
        df_demo.loc[(df_demo.score_zone.isin(['High Performance Zone', 'On-Track Zone'])) &
                    (df_demo.Status == 'low performing'), 'Status'] = 'top performing'
        
        df_demo.loc[(df_demo.score_zone.isin(['Unsustainable Zone', 'Off-Track Zone'])) &
                    (df_demo.Status == 'top performing'), 'Status'] = 'ahead of other drivers but still poor performance'
        
        # Assign question Status based on sentiment
        df_demo.loc[(df_demo['Employee Perception'].isin(['Very Positive', 'Positive'])) &
                    (df_demo.Type == 'Question'), 'Status'] = 'top performing'

        df_demo.loc[(df_demo['Employee Perception'].isin(['Very Negative', 'Negative'])) &
                    (df_demo.Type == 'Question'), 'Status'] = 'low performing'

        # Prepare question-only records: keep only flagged anomalies and required fields
        df_demo_quest = df_demo[df_demo.Type == 'Question'].reset_index(drop=True)
        df_demo_quest = df_demo_quest[(df_demo_quest.score_flag.notnull()) & (df_demo_quest.Type != 'Driver')][quest_col_names].reset_index(drop=True)

        # Prepare driver-only records: select top 2 and bottom 2 by rank
        df_demo_driver = df_demo[df_demo.Type == 'Driver'].reset_index(drop=True)
        top_driver_filter = df_demo_driver.sort_values('driver_rank', ascending=True).head(2)['driver_rank'].unique().tolist()   # Smallest ranks
        low_driver_filter = df_demo_driver.sort_values('driver_rank', ascending=False).head(2)['driver_rank'].unique().tolist()  # Largest ranks
        df_demo_driver = df_demo_driver[(df_demo_driver['driver_rank'].isin(top_driver_filter)) | (df_demo_driver['driver_rank'].isin(low_driver_filter))]  # Keep extremes

        # Cleanup driver records: order and drop non-informative rows
        df_demo_driver.sort_values(by=['Driver', 'employee_confidence_level'], inplace=True)  # Stable sorting for readability
        df_demo_driver.dropna(subset=['employee_confidence_level', 'score_flag', 'Employee Perception'], inplace=True, how='all')  # Remove rows with all key fields missing
        df_demo_driver = df_demo_driver[driver_col_names]  # Project only needed columns

        # Combine driver and question records, sorted to keep a logical order
        df_demo_all = pd.concat([df_demo_driver, df_demo_quest], ignore_index=True).sort_values(['driver_rank', 'Driver', 'Status'])
        # df_demo_all.drop(columns='driver_rank', inplace=True)  # Optionally drop rank if not needed downstream

        # Serialize to JSON then convert to Python objects for cleaning
        final_json = df_demo_all.to_json(orient='records')  # List-of-records orientation
        final_json = json.loads(final_json)                 # Deserialize to Python list
        final_json = self.clean_empty(final_json)           # Remove empty keys/values recursively
        
        return final_json  # Return cleaned list-of-dicts suitable for GenAI prompts

    def q1(self, x):
        """Calculate first quartile (25th percentile) for a numeric Series."""
        return x.quantile(0.25)  # Pandas quantile computation

    def q3(self, x):
        """Calculate third quartile (75th percentile) for a numeric Series."""
        return x.quantile(0.75)  # Pandas quantile computation

    def quantile_diff(self, train_data, metric):
        """
        Returns dataframe containing 1st and 3rd quantile of the surveys.
        """
        f = {metric: [self.q1, self.q3]}                  # Aggregations: apply q1 and q3 to metric
        quantile_df = train_data.groupby(["Survey"]).agg(f)  # Group by Survey and compute quantiles
        return quantile_df  # MultiIndex columns (metric -> q1/q3)
    
    def clean_empty(self, d):
        """
        clean empty/None values in json
        """
        if isinstance(d, dict):
            return {
                k: v 
                for k, v in ((k, self.clean_empty(v)) for k, v in d.items())
                if v  # Keep only keys with truthy values after cleaning
            }
        if isinstance(d, list):
            return [v for v in map(self.clean_empty, d) if v]  # Filter out falsy entries after cleaning
        return d  # Primitive types returned as-is
    
if __name__ == "__main__":
    survey = 2  # Example target survey cycle
    train_data = pd.read_csv("insights/test_output/transformation_sample_data.csv", index_col=False)  # Load sample dataset
    ob = transformation_summary()  # Instantiate processing class
    prompt = ob.get_prompt(train_data, survey)  # End-to-end pipeline result for survey
    convert_data = ob.convert_data(train_data)  # Intermediate: transformed DataFrame
    json_creation = ob.json_creation(convert_data, survey)  # Intermediate: cleaned JSON records
    diff = ob.quantile_diff(convert_data, "Score")  # Quantiles per survey for Score

    # print(prompt)  # Uncomment to inspect JSON payload
    # print(convert_data)  # Uncomment to inspect transformed DataFrame
    # print(json_creation)  # Uncomment to inspect cleaned JSON objects
    print(diff)  # Display quantile summary for verification