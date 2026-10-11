"""
Promoted Teams Mapping
Maps Championship seasons to the teams that were promoted to the Premier League
"""

# Championship Season → Promoted Teams (in order: Champions, Runners-up, Playoff winners)
PROMOTED_TEAMS = {
    "2019-2020": {
        "teams": ["Leeds", "West Brom", "Fulham"],
        "champions": "Leeds",
        "runners_up": "West Brom",
        "playoff_winners": "Fulham",
    },
    "2020-2021": {
        "teams": ["Norwich", "Watford", "Brentford"],
        "champions": "Norwich",
        "runners_up": "Watford",
        "playoff_winners": "Brentford",
    },
    "2021-2022": {
        "teams": ["Fulham", "Bournemouth", "Nott'm Forest"],
        "champions": "Fulham",
        "runners_up": "Bournemouth",
        "playoff_winners": "Nott'm Forest",
    },
    "2022-2023": {
        "teams": ["Burnley", "Sheffield Utd", "Luton"],
        "champions": "Burnley",
        "runners_up": "Sheffield Utd",
        "playoff_winners": "Luton",
    },
    "2023-2024": {
        "teams": ["Leicester", "Ipswich", "Southampton"],
        "champions": "Leicester",
        "runners_up": "Ipswich",
        "playoff_winners": "Southampton",
    },
    # 2024-2025 and 2025-2026 seasons are ongoing/future - will be updated
}

# Promotion context features for modeling
PROMOTION_TYPE_ENCODING = {
    "champions": 1,      # Dominant team, often regress less sharply
    "runners_up": 2,     # Strong team, moderate regression expected
    "playoff_winners": 3 # Won playoffs, often higher regression (stat inflation)
}

def get_promoted_teams(season):
    """Get list of teams promoted in a given season"""
    return PROMOTED_TEAMS.get(season, {}).get("teams", [])

def get_promotion_type(squad, season):
    """
    Get promotion context for a team in a season
    Returns: 'champions', 'runners_up', 'playoff_winners', or None
    """
    season_data = PROMOTED_TEAMS.get(season, {})
    if squad == season_data.get("champions"):
        return "champions"
    elif squad == season_data.get("runners_up"):
        return "runners_up"
    elif squad == season_data.get("playoff_winners"):
        return "playoff_winners"
    return None

def add_promotion_features(df):
    """
    Add promotion-related features to DataFrame
    
    Adds:
    - Was_Promoted: Binary (1 if promoted, 0 otherwise)
    - Promotion_Type: categorical (champions/runners_up/playoff_winners/None)
    - Promotion_Type_Code: numeric encoding (1/2/3/0)
    """
    import pandas as pd
    
    df['Was_Promoted'] = 0
    df['Promotion_Type'] = None
    df['Promotion_Type_Code'] = 0
    
    for season, data in PROMOTED_TEAMS.items():
        season_mask = df['Season'] == season
        
        for team in data['teams']:
            team_mask = season_mask & (df['Squad'] == team)
            df.loc[team_mask, 'Was_Promoted'] = 1
            
            promo_type = get_promotion_type(team, season)
            if promo_type:
                df.loc[team_mask, 'Promotion_Type'] = promo_type
                df.loc[team_mask, 'Promotion_Type_Code'] = PROMOTION_TYPE_ENCODING[promo_type]
    
    return df

# Example usage:
if __name__ == "__main__":
    import pandas as pd
    
    # Load EFL data
    df = pd.read_csv('efl_combined_all_cleaned.csv')
    
    # Add promotion features
    df = add_promotion_features(df)
    
    # Show results
    print("Promoted players by season:")
    print(df[df['Was_Promoted'] == 1].groupby(['Season', 'Squad']).size())
    
    print("\nPromotion type distribution:")
    print(df[df['Was_Promoted'] == 1]['Promotion_Type'].value_counts())
    
    # Save enhanced dataset
    df.to_csv('efl_combined_all_with_promotion.csv', index=False)
    print("\nSaved enhanced dataset with promotion features")
