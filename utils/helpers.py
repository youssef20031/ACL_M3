"""
Helper utility functions
"""
import re
from typing import List, Dict, Any, Optional


def format_player_stats(stats: Dict[str, Any]) -> str:
    """Format player statistics for display."""
    formatted = []
    for key, value in stats.items():
        # Convert snake_case to Title Case
        display_key = key.replace("_", " ").title()
        if isinstance(value, float):
            formatted.append(f"{display_key}: {value:.2f}")
        else:
            formatted.append(f"{display_key}: {value}")
    return "\n".join(formatted)


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching."""
    # Common abbreviations and variations
    team_aliases = {
        "man utd": "Man Utd",
        "manchester united": "Man Utd",
        "united": "Man Utd",
        "man city": "Man City",
        "manchester city": "Man City",
        "city": "Man City",
        "spurs": "Spurs",
        "tottenham": "Spurs",
        "tottenham hotspur": "Spurs",
        "nottingham forest": "Nott'm Forest",
        "forest": "Nott'm Forest",
        "wolves": "Wolves",
        "wolverhampton": "Wolves",
        "brighton": "Brighton",
        "brighton & hove albion": "Brighton",
        "west ham": "West Ham",
        "west ham united": "West Ham",
        "newcastle": "Newcastle",
        "newcastle united": "Newcastle",
        "aston villa": "Aston Villa",
        "villa": "Aston Villa",
        "crystal palace": "Crystal Palace",
        "palace": "Crystal Palace",
    }
    
    normalized = name.lower().strip()
    return team_aliases.get(normalized, name.title())


def normalize_position(position: str) -> str:
    """Normalize position code."""
    position_map = {
        "goalkeeper": "GK",
        "gk": "GK",
        "keeper": "GK",
        "defender": "DEF",
        "def": "DEF",
        "defense": "DEF",
        "midfielder": "MID",
        "mid": "MID",
        "midfield": "MID",
        "forward": "FWD",
        "fwd": "FWD",
        "striker": "FWD",
        "attacker": "FWD",
    }
    
    normalized = position.lower().strip()
    return position_map.get(normalized, position.upper())


def extract_numbers(text: str) -> List[int]:
    """Extract all numbers from text."""
    return [int(n) for n in re.findall(r'\d+', text)]


def format_value(value: int) -> str:
    """Format player value in millions."""
    return f"£{value / 10:.1f}m"


def format_large_number(num: int) -> str:
    """Format large numbers with K/M suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def clean_player_name(name: str) -> str:
    """Clean and normalize player name."""
    # Remove extra whitespace
    name = " ".join(name.split())
    # Title case
    return name.title()


def calculate_points_per_million(total_points: int, value: int) -> float:
    """Calculate points per million value."""
    if value == 0:
        return 0.0
    return (total_points / (value / 10))


def calculate_points_per_game(total_points: int, games: int) -> float:
    """Calculate points per game."""
    if games == 0:
        return 0.0
    return total_points / games


def get_season_from_date(date_str: str) -> str:
    """Determine season from kickoff date."""
    from datetime import datetime
    
    if isinstance(date_str, str):
        # Parse date
        date = datetime.fromisoformat(date_str.replace("+00:00", ""))
    else:
        date = date_str
    
    year = date.year
    month = date.month
    
    # Season starts in August
    if month >= 8:
        return f"{year}-{str(year + 1)[2:]}"
    else:
        return f"{year - 1}-{str(year)[2:]}"


def create_player_description(player_data: Dict[str, Any]) -> str:
    """
    Create a text description of a player for embedding.
    
    Args:
        player_data: Dictionary with player information
        
    Returns:
        Text description for embedding
    """
    name = player_data.get("name", "Unknown")
    position = player_data.get("position", "")
    team = player_data.get("team", "")
    season = player_data.get("season", "")
    
    # Stats
    goals = player_data.get("total_goals", 0)
    assists = player_data.get("total_assists", 0)
    points = player_data.get("total_points", 0)
    clean_sheets = player_data.get("total_clean_sheets", 0)
    bonus = player_data.get("total_bonus", 0)
    
    # Build description
    position_name = {
        "GK": "goalkeeper",
        "DEF": "defender", 
        "MID": "midfielder",
        "FWD": "forward"
    }.get(position, "player")
    
    description = f"{name} is a {position_name}"
    
    if team:
        description += f" who plays for {team}"
    
    if season:
        description += f" in the {season} season"
    
    description += f". Total points: {points}."
    
    if position in ["FWD", "MID"]:
        description += f" Goals: {goals}, assists: {assists}."
    elif position == "DEF":
        description += f" Clean sheets: {clean_sheets}, assists: {assists}."
    elif position == "GK":
        description += f" Clean sheets: {clean_sheets}."
    
    if bonus > 0:
        description += f" Bonus points: {bonus}."
    
    return description


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
