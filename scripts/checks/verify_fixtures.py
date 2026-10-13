"""
Verify the Premier League 2026-27 Fixtures CSV
"""

import csv
from collections import Counter

def verify_fixtures(csv_file='data/FPL_2026_2027_fixtures.csv'):
    print("="*60)
    print("Verifying Premier League 2026-27 Fixtures")
    print("="*60)
    print()
    
    fixtures = []
    teams = set()
    matchdays = set()
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fixtures.append(row)
            teams.add(row['Home Team'])
            teams.add(row['Away Team'])
            matchdays.add(int(row['Matchday']))
    
    # Basic counts
    print(f"✓ Total fixtures: {len(fixtures)}")
    print(f"✓ Total teams: {len(teams)}")
    print(f"✓ Total matchdays: {len(matchdays)}")
    print()
    
    # Expected values for 20-team league
    expected_fixtures = 380  # 20 teams * 19 matchdays * 2 (home/away)
    expected_matchdays = 38
    expected_teams = 20
    
    # Validation
    errors = []
    
    if len(fixtures) != expected_fixtures:
        errors.append(f"✗ Expected {expected_fixtures} fixtures, got {len(fixtures)}")
    else:
        print(f"✓ Correct number of fixtures ({expected_fixtures})")
    
    if len(teams) != expected_teams:
        errors.append(f"✗ Expected {expected_teams} teams, got {len(teams)}")
    else:
        print(f"✓ Correct number of teams ({expected_teams})")
    
    if len(matchdays) != expected_matchdays:
        errors.append(f"✗ Expected {expected_matchdays} matchdays, got {len(matchdays)}")
    else:
        print(f"✓ Correct number of matchdays ({expected_matchdays})")
    
    # Check first fixture
    first = fixtures[0]
    if first['Home Team'] == 'Arsenal' and first['Away Team'] == 'Coventry City' and '21/08/2026' in first['Date']:
        print(f"✓ First fixture confirmed: Arsenal vs Coventry City on 21/08/2026")
    else:
        errors.append(f"✗ First fixture incorrect: {first}")
    
    # Check fixtures per matchday
    fixtures_per_matchday = Counter(int(f['Matchday']) for f in fixtures)
    incorrect_matchdays = [md for md, count in fixtures_per_matchday.items() if count != 10]
    
    if not incorrect_matchdays:
        print(f"✓ All matchdays have exactly 10 fixtures")
    else:
        for md in incorrect_matchdays:
            errors.append(f"✗ Matchday {md} has {fixtures_per_matchday[md]} fixtures (expected 10)")
    
    # List all teams
    print(f"\nTeams in dataset:")
    for i, team in enumerate(sorted(teams), 1):
        print(f"  {i}. {team}")
    
    # Summary
    print()
    print("="*60)
    if errors:
        print("VALIDATION FAILED")
        print("="*60)
        for error in errors:
            print(error)
        return False
    else:
        print("✓ ALL VALIDATIONS PASSED")
        print("="*60)
        print("\nThe fixture data is correct and ready to use!")
        return True


if __name__ == "__main__":
    verify_fixtures()
