"""Process fixtures CSV to add required columns."""
import pandas as pd
from datetime import datetime

# Load fixtures
fixtures = pd.read_csv('FPL_2026_2027_fixtures_fixed.csv')

# Rename columns to match expected format
fixtures = fixtures.rename(columns={
    'Matchday': 'gameweek',
    'Home Team': 'home_team',
    'Away Team': 'away_team'
})

# Combine Date and Time into kickoff_time
fixtures['kickoff_time'] = pd.to_datetime(
    fixtures['Date'] + ' ' + fixtures['Time'], 
    format='%d/%m/%Y %H:%M'
).dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Keep only needed columns
fixtures = fixtures[['gameweek', 'kickoff_time', 'home_team', 'away_team']]

# Save processed fixtures
fixtures.to_csv('FPL_2026_2027_fixtures_processed.csv', index=False)

print(f"✅ Processed {len(fixtures)} fixtures")
print(f"   Gameweeks: {sorted(fixtures['gameweek'].unique())}")
print(f"\nSample:")
print(fixtures.head())
