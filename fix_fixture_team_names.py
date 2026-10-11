"""Fix team names in fixtures CSV to match Neo4j database"""
import pandas as pd

# Mapping from CSV names to Neo4j database names
TEAM_NAME_MAPPING = {
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Tottenham Hotspur": "Spurs",
    "Brighton and Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Hull City": "Hull City",  # Not in DB, will skip
    "Coventry City": "Coventry City",  # Not in DB, will skip
}

def fix_team_names():
    """Fix team names in the fixtures CSV."""
    
    # Read the CSV
    df = pd.read_csv('FPL_2026_2027_fixtures.csv')
    
    print(f"Original CSV has {len(df)} fixtures")
    print(f"\nOriginal unique teams:")
    all_teams = set(df['Home Team'].unique()) | set(df['Away Team'].unique())
    for team in sorted(all_teams):
        print(f"  - {team}")
    
    # Apply mapping
    df['Home Team'] = df['Home Team'].replace(TEAM_NAME_MAPPING)
    df['Away Team'] = df['Away Team'].replace(TEAM_NAME_MAPPING)
    
    print(f"\nAfter mapping:")
    all_teams = set(df['Home Team'].unique()) | set(df['Away Team'].unique())
    for team in sorted(all_teams):
        print(f"  - {team}")
    
    # Save fixed version
    output_file = 'FPL_2026_2027_fixtures_fixed.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved fixed fixtures to {output_file}")
    print(f"Total fixtures: {len(df)}")
    
    return output_file

if __name__ == "__main__":
    fix_team_names()
