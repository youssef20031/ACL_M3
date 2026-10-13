import pandas as pd

df = pd.read_csv('data/cleaned_merged_seasons_cleaned.csv')
gw29 = df[df['GW'] == 29]
print(f'Players in GW29: {gw29["name"].nunique()}')
print(f'Seasons in GW29: {sorted(gw29["season_x"].unique())}')
print(f'Total records in GW29: {len(gw29)}')

# Get most recent GW29 (latest season)
latest_season = gw29['season_x'].max()
latest_gw29 = gw29[gw29['season_x'] == latest_season]
print(f'\nLatest season GW29 ({latest_season}):')
print(f'  Players: {latest_gw29["name"].nunique()}')
print(f'  Records: {len(latest_gw29)}')

# Save active players list
active_players = latest_gw29[['name', 'position', 'team_x']].drop_duplicates()
print(f'\nActive players: {len(active_players)}')
active_players.to_csv('data/active_players_gw29.csv', index=False)
print('Saved to data/active_players_gw29.csv')
