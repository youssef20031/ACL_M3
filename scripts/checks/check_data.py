import pandas as pd

df = pd.read_csv('data/cleaned_merged_seasons_cleaned.csv')
print(f'Total records: {len(df):,}')
print(f'Seasons: {sorted(df["season_x"].unique())}')
print(f'Date range: {df["kickoff_time"].min()} to {df["kickoff_time"].max()}')
print(f'\nColumns: {list(df.columns)}')
