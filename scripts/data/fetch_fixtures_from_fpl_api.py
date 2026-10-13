"""
Fetch Premier League 2026-27 Fixtures from FPL API
Uses the official Fantasy Premier League API to get fixture data
"""

import requests
import csv
from datetime import datetime
import json

# FPL API endpoints
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
TEAMS_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_teams():
    """Fetch team data to map team IDs to names."""
    print("Fetching team data from FPL API...")
    response = requests.get(TEAMS_URL)
    
    if response.status_code != 200:
        print(f"✗ Failed to fetch teams: HTTP {response.status_code}")
        return None
    
    data = response.json()
    teams = {team['id']: team['name'] for team in data['teams']}
    
    print(f"✓ Found {len(teams)} teams")
    return teams


def fetch_fixtures():
    """Fetch fixture data from FPL API."""
    print("Fetching fixtures from FPL API...")
    response = requests.get(FIXTURES_URL)
    
    if response.status_code != 200:
        print(f"✗ Failed to fetch fixtures: HTTP {response.status_code}")
        return None
    
    fixtures = response.json()
    print(f"✓ Found {len(fixtures)} fixtures")
    return fixtures


def convert_to_csv(fixtures, teams, output_file='data/FPL_2026_2027_fixtures.csv'):
    """Convert fixtures to CSV format."""
    
    if not fixtures or not teams:
        print("✗ No data to convert")
        return False
    
    csv_fixtures = []
    
    for fixture in fixtures:
        # Skip if fixture doesn't have required data
        if not fixture.get('team_h') or not fixture.get('team_a') or not fixture.get('event'):
            continue
        
        # Get team names
        home_team = teams.get(fixture['team_h'], f"Team {fixture['team_h']}")
        away_team = teams.get(fixture['team_a'], f"Team {fixture['team_a']}")
        
        # Parse kickoff time
        kickoff_time_str = fixture.get('kickoff_time')
        if kickoff_time_str:
            try:
                # Parse ISO format: "2026-08-21T19:00:00Z"
                kickoff_dt = datetime.fromisoformat(kickoff_time_str.replace('Z', '+00:00'))
                date_str = kickoff_dt.strftime("%d/%m/%Y")
                time_str = kickoff_dt.strftime("%H:%M")
            except:
                date_str = "TBD"
                time_str = "15:00"
        else:
            date_str = "TBD"
            time_str = "15:00"
        
        # Matchday (called 'event' in FPL API)
        matchday = fixture.get('event', 0)
        
        csv_fixtures.append({
            'Matchday': matchday,
            'Date': date_str,
            'Time': time_str,
            'Home Team': home_team,
            'Away Team': away_team
        })
    
    # Sort by matchday and date
    csv_fixtures.sort(key=lambda x: (x['Matchday'], x['Date']))
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Matchday', 'Date', 'Time', 'Home Team', 'Away Team']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for fixture in csv_fixtures:
            writer.writerow(fixture)
    
    print(f"✓ Created {output_file} with {len(csv_fixtures)} fixtures")
    
    # Verify first fixture
    if csv_fixtures:
        first = csv_fixtures[0]
        print(f"\nFirst fixture verification:")
        print(f"  Date: {first['Date']} {first['Time']}")
        print(f"  Match: {first['Home Team']} vs {first['Away Team']}")
        
        # Check if it's Arsenal vs Coventry on 21st August
        if "Arsenal" in first['Home Team'] and "Coventry" in first['Away Team']:
            print(f"  ✓ Confirmed: This is the 2026-27 season!")
        else:
            print(f"  ⚠ Warning: First fixture doesn't match expected (Arsenal vs Coventry)")
    
    return True


def main():
    print("="*60)
    print("FPL API Fixture Fetcher - Season 2026/27")
    print("="*60)
    print()
    
    # Fetch teams
    teams = fetch_teams()
    if not teams:
        print("\n✗ Failed to fetch team data")
        return 1
    
    # Fetch fixtures
    fixtures = fetch_fixtures()
    if not fixtures:
        print("\n✗ Failed to fetch fixture data")
        return 1
    
    # Convert to CSV
    success = convert_to_csv(fixtures, teams)
    
    if success:
        print("\n✓ Successfully created fixture CSV file!")
        return 0
    else:
        print("\n✗ Failed to create CSV file")
        return 1


if __name__ == "__main__":
    exit(main())
