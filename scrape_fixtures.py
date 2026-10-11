"""
Web Scraper for Premier League 2026-27 Fixtures
Extracts fixture data from the scraped content and creates a CSV file
"""

import csv
import re
from datetime import datetime

# Scraped fixture data from the Premier League website
FIXTURE_TEXT = """
Friday 21 August 202620:00 Arsenal v Coventry City (Sky Sports)
Saturday 22 August12:30 Hull City v Manchester United (TNT Sports)Everton v Crystal PalaceIpswich Town v SunderlandNottingham Forest v Leeds United17:30 Brentford v Tottenham Hotspur (Sky Sports)
Sunday 23 August14:00 Brighton & Hove Albion v Aston Villa (Sky Sports)14:00 Manchester City v AFC Bournemouth (Sky Sports)16:30 Newcastle United v Liverpool (Sky Sports)
Monday 24 August20:00 Fulham v Chelsea (Sky Sports)
Saturday 29 AugustAFC Bournemouth v EvertonAston Villa v ArsenalChelsea v Brighton & Hove AlbionCoventry City v Hull CityCrystal Palace v Manchester CityLeeds United v BrentfordLiverpool v Nottingham ForestManchester United v Ipswich TownSunderland v FulhamTottenham Hotspur v Newcastle United
Saturday 5 SeptemberArsenal v ChelseaBrentford v SunderlandBrighton & Hove Albion v Leeds UnitedEverton v Manchester UnitedFulham v Crystal PalaceHull City v Aston VillaIpswich Town v LiverpoolManchester City v Coventry CityNewcastle United v AFC BournemouthNottingham Forest v Tottenham Hotspur
Saturday 12 SeptemberAFC Bournemouth v BrentfordAston Villa v Nottingham ForestChelsea v Hull CityCoventry City v Brighton & Hove AlbionCrystal Palace v Ipswich TownLeeds United v Newcastle UnitedLiverpool v FulhamManchester United v Manchester CitySunderland v ArsenalTottenham Hotspur v Everton
Saturday 19 SeptemberAFC Bournemouth v LiverpoolBrentford v ChelseaBrighton & Hove Albion v ArsenalEverton v Ipswich TownFulham v Manchester UnitedLeeds United v Crystal PalaceManchester City v SunderlandNewcastle United v Hull CityNottingham Forest v Coventry CityTottenham Hotspur v Aston Villa
Saturday 10 OctoberArsenal v Leeds UnitedAston Villa v BrentfordChelsea v AFC BournemouthCoventry City v Newcastle UnitedCrystal Palace v Nottingham ForestHull City v EvertonIpswich Town v FulhamLiverpool v Manchester CityManchester United v Tottenham HotspurSunderland v Brighton & Hove Albion
Saturday 17 OctoberAFC Bournemouth v SunderlandBrentford v LiverpoolBrighton & Hove Albion v Crystal PalaceEverton v ChelseaFulham v Hull CityLeeds United v Manchester UnitedManchester City v Ipswich TownNewcastle United v Aston VillaNottingham Forest v ArsenalTottenham Hotspur v Coventry City
Saturday 24 OctoberArsenal v EvertonAston Villa v Manchester CityChelsea v Tottenham HotspurCoventry City v FulhamCrystal Palace v Newcastle UnitedHull City v BrentfordIpswich Town v Nottingham ForestLiverpool v Brighton & Hove AlbionManchester United v AFC BournemouthSunderland v Leeds United
Saturday 31 OctoberAFC Bournemouth v Leeds UnitedAston Villa v FulhamBrentford v Nottingham ForestChelsea v Manchester UnitedCoventry City v SunderlandHull City v Ipswich TownLiverpool v ArsenalManchester City v Brighton & Hove AlbionNewcastle United v EvertonTottenham Hotspur v Crystal Palace
Saturday 7 November Arsenal v Hull CityBrighton & Hove Albion v BrentfordCrystal Palace v LiverpoolEverton v Coventry CityFulham v Newcastle UnitedIpswich Town v AFC BournemouthLeeds United v Tottenham HotspurManchester United v Aston VillaNottingham Forest v Manchester CitySunderland v Chelsea
Saturday 21 November AFC Bournemouth v Nottingham ForestAston Villa v SunderlandBrentford v EvertonChelsea v Leeds UnitedCoventry City v Crystal PalaceHull City v Brighton & Hove AlbionLiverpool v Manchester UnitedManchester City v FulhamNewcastle United v ArsenalTottenham Hotspur v Ipswich Town
Saturday 28 November Arsenal v Manchester CityBrighton & Hove Albion v Newcastle UnitedCrystal Palace v Hull CityEverton v LiverpoolFulham v AFC BournemouthIpswich Town v Aston VillaLeeds United v Coventry CityManchester United v BrentfordNottingham Forest v ChelseaSunderland v Tottenham Hotspur
Wednesday 2 December20:00 AFC Bournemouth v Brighton & Hove Albion20:00 Aston Villa v Everton 20:00 Brentford v Arsenal 20:00 Chelsea v Crystal Palace 20:00 Coventry City v Ipswich Town 20:00 Hull City v Nottingham Forest20:00 Liverpool v Sunderland 20:00 Manchester City v Leeds United 20:00 Newcastle United v Manchester United20:00 Tottenham Hotspur v Fulham 
Saturday 5 DecemberAFC Bournemouth v Hull CityAston Villa v Crystal PalaceBrentford v Manchester CityChelsea v LiverpoolEverton v FulhamLeeds United v Ipswich TownManchester United v Coventry CityNewcastle United v SunderlandNottingham Forest v Brighton & Hove AlbionTottenham Hotspur v Arsenal
Saturday 12 DecemberArsenal v AFC BournemouthBrighton & Hove Albion v EvertonCoventry City v Aston VillaCrystal Palace v Manchester UnitedFulham v BrentfordHull City v Tottenham HotspurIpswich Town v Newcastle UnitedLiverpool v Leeds UnitedManchester City v ChelseaSunderland v Nottingham Forest
Saturday 19 DecemberAFC Bournemouth v Coventry CityArsenal v Manchester UnitedBrentford v Newcastle UnitedBrighton & Hove Albion v Ipswich TownChelsea v Aston VillaLeeds United v FulhamLiverpool v Tottenham HotspurManchester City v Hull CityNottingham Forest v EvertonSunderland v Crystal Palace
Saturday 26 DecemberAston Villa v Leeds UnitedCoventry City v ChelseaCrystal Palace v ArsenalEverton v SunderlandFulham v Brighton & Hove AlbionHull City v LiverpoolIpswich Town v BrentfordManchester United v Nottingham ForestNewcastle United v Manchester CityTottenham Hotspur v AFC Bournemouth
Wednesday 30 December 20:00 Aston Villa v Liverpool 20:00 Coventry City v Brentford 20:00 Crystal Palace v AFC Bournemouth20:00 Everton v Manchester City20:00 Fulham v Arsenal 20:00 Hull City v Leeds United20:00 Ipswich Town v Chelsea 20:00 Manchester United v Sunderland 20:00 Newcastle United v Nottingham Forest 20:00 Tottenham Hotspur v Brighton & Hove Albion 
Saturday 2 January 2027AFC Bournemouth v Aston VillaArsenal v Ipswich TownBrentford v Crystal PalaceBrighton & Hove Albion v Manchester UnitedChelsea v Newcastle UnitedLeeds United v EvertonLiverpool v Coventry CityManchester City v Tottenham HotspurNottingham Forest v FulhamSunderland v Hull City
Wednesday 6 January20:00 Arsenal v Brentford20:00 Brighton & Hove Albion v AFC Bournemouth20:00 Crystal Palace v Chelsea 20:00 Everton v Aston Villa 20:00 Fulham v Tottenham Hotspur20:00 Ipswich Town v Coventry City20:00 Leeds United v Manchester City20:00 Manchester United v Newcastle United20:00 Nottingham Forest v Hull City 20:00 Sunderland v Liverpool 
Saturday 16 JanuaryAFC Bournemouth v Ipswich TownAston Villa v Manchester UnitedBrentford v Brighton & Hove AlbionChelsea v SunderlandCoventry City v EvertonHull City v ArsenalLiverpool v Crystal PalaceManchester City v Nottingham ForestNewcastle United v FulhamTottenham Hotspur v Leeds United
Saturday 23 JanuaryArsenal v Newcastle UnitedBrighton & Hove Albion v Manchester CityCrystal Palace v Tottenham HotspurEverton v BrentfordFulham v Aston VillaIpswich Town v Hull CityLeeds United v ChelseaManchester United v LiverpoolNottingham Forest v AFC BournemouthSunderland v Coventry City
Saturday 30 JanuaryAFC Bournemouth v FulhamAston Villa v Ipswich TownBrentford v Manchester UnitedChelsea v Nottingham ForestCoventry City v Leeds UnitedHull City v Crystal PalaceLiverpool v EvertonManchester City v ArsenalNewcastle United v Brighton & Hove AlbionTottenham Hotspur v Sunderland
Saturday 6 FebruaryArsenal v LiverpoolBrighton & Hove Albion v Hull CityCrystal Palace v Coventry CityEverton v Newcastle UnitedFulham v Manchester CityIpswich Town v Tottenham HotspurLeeds United v AFC BournemouthManchester United v ChelseaNottingham Forest v BrentfordSunderland v Aston Villa
Wednesday 10 February20:00 Aston Villa v AFC Bournemouth20:00 Coventry City v Liverpool 20:00 Crystal Palace v Brentford20:00 Everton v Leeds United 20:00 Fulham v Nottingham Forest20:00 Hull City v Sunderland 20:00 Ipswich Town v Arsenal 20:00 Manchester United v Brighton & Hove Albion 20:00 Newcastle United v Chelsea 20:00 Tottenham Hotspur v Manchester City 
Saturday 20 FebruaryAFC Bournemouth v Crystal PalaceArsenal v FulhamBrentford v Coventry CityBrighton & Hove Albion v Tottenham HotspurChelsea v Ipswich TownLeeds United v Aston VillaLiverpool v Hull CityManchester City v Newcastle UnitedNottingham Forest v Manchester UnitedSunderland v Everton
Saturday 27 FebruaryAston Villa v ChelseaCoventry City v AFC BournemouthCrystal Palace v SunderlandEverton v Nottingham ForestFulham v Leeds UnitedHull City v Manchester CityIpswich Town v Brighton & Hove AlbionManchester United v ArsenalNewcastle United v BrentfordTottenham Hotspur v Liverpool
Wednesday 3 March20:00 AFC Bournemouth v Tottenham Hotspur20:00 Arsenal v Crystal Palace20:00 Brentford v Ipswich Town 20:00 Brighton & Hove Albion v Fulham20:00 Chelsea v Coventry City20:00 Leeds United v Hull City20:00 Liverpool v Aston Villa20:00 Manchester City v Everton20:00 Nottingham Forest v Newcastle United20:00 Sunderland v Manchester United 
Saturday 13 MarchAFC Bournemouth v Newcastle UnitedAston Villa v Hull CityChelsea v ArsenalCoventry City v Manchester CityCrystal Palace v FulhamLeeds United v Brighton & Hove AlbionLiverpool v Ipswich TownManchester United v EvertonSunderland v BrentfordTottenham Hotspur v Nottingham Forest
Saturday 20 MarchArsenal v SunderlandBrentford v AFC BournemouthBrighton & Hove Albion v Coventry CityEverton v Tottenham HotspurFulham v LiverpoolHull City v ChelseaIpswich Town v Crystal PalaceManchester City v Manchester UnitedNewcastle United v Leeds UnitedNottingham Forest v Aston Villa
Saturday 10 April AFC Bournemouth v Manchester CityAston Villa v Brighton & Hove AlbionChelsea v FulhamCoventry City v ArsenalCrystal Palace v EvertonLeeds United v Nottingham ForestLiverpool v Newcastle UnitedManchester United v Hull CitySunderland v Ipswich TownTottenham Hotspur v Brentford
Saturday 17 AprilArsenal v Aston VillaBrentford v Leeds UnitedBrighton & Hove Albion v ChelseaEverton v AFC BournemouthFulham v SunderlandHull City v Coventry CityIpswich Town v Manchester UnitedManchester City v Crystal PalaceNewcastle United v Tottenham HotspurNottingham Forest v Liverpool
Saturday 24 AprilAFC Bournemouth v ArsenalAston Villa v Coventry CityBrentford v FulhamChelsea v Manchester CityEverton v Brighton & Hove AlbionLeeds United v LiverpoolManchester United v Crystal PalaceNewcastle United v Ipswich TownNottingham Forest v SunderlandTottenham Hotspur v Hull City
Saturday 1 May Arsenal v Tottenham HotspurBrighton & Hove Albion v Nottingham ForestCoventry City v Manchester UnitedCrystal Palace v Aston VillaFulham v EvertonHull City v AFC BournemouthIpswich Town v Leeds UnitedLiverpool v ChelseaManchester City v BrentfordSunderland v Newcastle United
Saturday 8 MayAFC Bournemouth v Manchester UnitedBrentford v Aston VillaBrighton & Hove Albion v SunderlandEverton v Hull CityFulham v Ipswich TownLeeds United v ArsenalManchester City v LiverpoolNewcastle United v Coventry CityNottingham Forest v Crystal PalaceTottenham Hotspur v Chelsea
Saturday 15 MayArsenal v Nottingham ForestAston Villa v Newcastle UnitedChelsea v EvertonCoventry City v Tottenham HotspurCrystal Palace v Brighton & Hove AlbionHull City v FulhamIpswich Town v Manchester CityLiverpool v BrentfordManchester United v Leeds UnitedSunderland v AFC Bournemouth
Sunday 23 MayAFC Bournemouth v ChelseaBrentford v Hull CityBrighton & Hove Albion v LiverpoolEverton v ArsenalFulham v Coventry CityLeeds United v SunderlandManchester City v Aston VillaNewcastle United v Crystal PalaceNottingham Forest v Ipswich TownTottenham Hotspur v Manchester United
Sunday 30 MayArsenal v Brighton & Hove AlbionAston Villa v Tottenham HotspurChelsea v BrentfordCoventry City v Nottingham ForestCrystal Palace v Leeds UnitedHull City v Newcastle UnitedIpswich Town v EvertonLiverpool v AFC BournemouthManchester United v FulhamSunderland v Manchester City
"""

# Team name normalization (full names as they appear in Neo4j)
TEAM_NAMES = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry City",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull City": "Hull City",
    "Ipswich Town": "Ipswich Town",
    "Leeds United": "Leeds United",
    "Liverpool": "Liverpool",
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Newcastle United": "Newcastle United",
    "Nottingham Forest": "Nottingham Forest",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham Hotspur",
}


def parse_fixtures():
    """Parse the fixture text and extract structured data."""
    fixtures = []
    matchday = 0
    
    # Split by lines and parse each matchday
    text = FIXTURE_TEXT.strip()
    
    # Match patterns for date + fixtures blocks
    # Pattern: Day Month (Year)? followed by fixtures
    matchday_blocks = re.split(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d+\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?)', text)
    
    current_date = None
    
    for i in range(1, len(matchday_blocks), 2):
        if i >= len(matchday_blocks):
            break
            
        date_str = matchday_blocks[i].strip()
        fixtures_text = matchday_blocks[i + 1] if i + 1 < len(matchday_blocks) else ""
        
        # Parse date
        date_match = re.match(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\d+)\s+(January|February|March|April|May|June|July|August|September|October|November|December)(\s+(\d{4}))?', date_str)
        if date_match:
            day_num = date_match.group(2)
            month = date_match.group(3)
            year = date_match.group(5) if date_match.group(5) else ("2027" if month in ["January", "February", "March", "April", "May"] else "2026")
            
            current_date = f"{day_num.zfill(2)}/{month[:3]}/{year}"
        
        if not current_date:
            continue
        
        # Extract time and fixtures
        # Split by time markers or fixtures
        lines = fixtures_text.split('\n')
        combined_text = ' '.join(lines)
        
        # Find all time:fixture patterns
        # Pattern: optional time (HH:MM) followed by "Team v Team"
        pattern = r'(?:(\d{1,2}):(\d{2})\s+)?([A-Za-z\s&]+?)\s+v\s+([A-Za-z\s&]+?)(?=\s*\(|$|\d{1,2}:|\s+[A-Z][a-z]+\s+[A-Z])'
        
        matches = re.finditer(pattern, combined_text)
        
        current_time = "15:00"
        day_fixtures = []
        
        for match in matches:
            if match.group(1) and match.group(2):
                current_time = f"{match.group(1).zfill(2)}:{match.group(2)}"
            
            home = match.group(3).strip()
            away = match.group(4).strip()
            
            # Remove trailing punctuation/extras
            home = re.sub(r'\s*\(.*$', '', home).strip()
            away = re.sub(r'\s*\(.*$', '', away).strip()
            
            # Normalize
            home_normalized = TEAM_NAMES.get(home, home)
            away_normalized = TEAM_NAMES.get(away, away)
            
            if home_normalized in TEAM_NAMES.values() and away_normalized in TEAM_NAMES.values():
                day_fixtures.append({
                    'Date': current_date,
                    'Time': current_time,
                    'Home Team': home_normalized,
                    'Away Team': away_normalized
                })
        
        # If we got fixtures for this date, increment matchday
        if day_fixtures:
            # Check if it's a new matchday (more than just a few fixtures on same date)
            if len(day_fixtures) >= 5 or not fixtures or fixtures[-1]['Date'] != current_date:
                matchday += 1
            
            for fixture in day_fixtures:
                fixture['Matchday'] = matchday
                fixtures.append(fixture)
    
    return fixtures


def create_csv(fixtures, output_file='FPL_2026_2027_fixtures.csv'):
    """Create CSV file from parsed fixtures."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Matchday', 'Date', 'Time', 'Home Team', 'Away Team']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for fixture in fixtures:
            writer.writerow(fixture)
    
    print(f"✓ Created {output_file} with {len(fixtures)} fixtures")


if __name__ == "__main__":
    print("Parsing Premier League 2026-27 fixtures...")
    fixtures = parse_fixtures()
    print(f"Found {len(fixtures)} fixtures")
    
    create_csv(fixtures)
    print("Done!")
