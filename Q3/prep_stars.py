import pandas as pd
import numpy as np

class StarPreprocessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.raw_df = None

    def load_data(self):
        # 1. Load data
        self.raw_df = pd.read_csv(self.csv_path, parse_dates=['timestamp'])
        self.raw_df.columns = self.raw_df.columns.str.strip()
        
        # 2. Sort by time
        self.raw_df = self.raw_df.sort_values('timestamp')
        
        # 3. Create a Date column (without time) to identify daily buckets
        self.raw_df['date_only'] = self.raw_df['timestamp'].dt.date
        
        # 4. Drop Duplicates Strategy:
        # We keep the 'last' entry for every (Repo, Date) pair.
        initial_count = len(self.raw_df)
        self.raw_df = self.raw_df.drop_duplicates(
            subset=['repository_id', 'date_only'], 
            keep='last'
        )
        
        self.raw_df = self.raw_df.drop(columns=['date_only'])

    def get_repo_series(self, repo_name):
        repo_df = self.raw_df[self.raw_df['repository_id'] == repo_name].copy()
        
        if repo_df.empty:
            available = self.raw_df['repository_id'].unique()[:5]
            raise ValueError(f"Repository '{repo_name}' not found. Available: {available}...")

        repo_df = repo_df.set_index('timestamp')
        daily_series = repo_df['stars'].resample('D').max()
        
        # handle missing days
        daily_series = daily_series.ffill()
        
        # handle start gaps
        daily_series = daily_series.fillna(0)

        # iincremental difference
        incremental_series = daily_series.diff().fillna(0)
        return daily_series, incremental_series

    def normalize_series(self, train_data, test_data):
        min_val = train_data.min()
        max_val = train_data.max()
        
        if max_val == min_val:
            return train_data, test_data, min_val, max_val

        train_scaled = (train_data - min_val) / (max_val - min_val)
        test_scaled = (test_data - min_val) / (max_val - min_val)
        
        return train_scaled, test_scaled, min_val, max_val

    @staticmethod
    def inverse_transform(scaled_data, min_val, max_val):
        return scaled_data * (max_val - min_val) + min_val