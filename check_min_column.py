import pandas as pd

df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_all_cleaned.csv')
print('Min column after cleaning:')
print('Dtype:', df['Min'].dtype)
print('Null count:', df['Min'].isna().sum(), '/', len(df))
print('Non-null sample:')
print(df[df['Min'].notna()][['Player', 'Season', 'Min', '90s']].head(20))

# Check what columns contain minutes
min_cols = [col for col in df.columns if 'min' in col.lower()]
print(f'\nColumns containing "min": {min_cols}')

# The problem: Min from standard table doesn't exist - it came from playing_time!
# Check original file
print('\nChecking original standard table structure...')
from efl_data_ingestion import load_fbref_table
std_df = load_fbref_table('EFL_Data_All_Seasons/EFL-2019-2020/2019-2020_Standard_Players.txt', 
                           season='2019-2020', table_type='standard')
print('Standard table columns containing "min":', [col for col in std_df.columns if 'min' in col.lower()])

pt_df = load_fbref_table('EFL_Data_All_Seasons/EFL-2019-2020/2019-2020_Playtime_Players.txt',
                         season='2019-2020', table_type='playing_time')
print('Playing time table columns containing "min":', [col for col in pt_df.columns if 'min' in col.lower()])
