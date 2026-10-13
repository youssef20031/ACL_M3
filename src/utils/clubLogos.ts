/**
 * Club logos and abbreviations for FPL teams
 */

export interface ClubInfo {
  abbreviation: string;
  logoPath: string;
}

// Mapping of team names to their abbreviations and logo paths
export const CLUB_MAPPING: Record<string, ClubInfo> = {
  'Arsenal': { abbreviation: 'ARS', logoPath: '/logos/Arsenal.png' },
  'Aston Villa': { abbreviation: 'AVL', logoPath: '/logos/Aston_Villa.png' },
  'Bournemouth': { abbreviation: 'BOU', logoPath: '/logos/Bournemouth.png' },
  'Brentford': { abbreviation: 'BRE', logoPath: '/logos/Brentford.png' },
  'Brighton': { abbreviation: 'BHA', logoPath: '/logos/Brighton.png' },
  'Chelsea': { abbreviation: 'CHE', logoPath: '/logos/Chelsea.png' },
  'Crystal Palace': { abbreviation: 'CRY', logoPath: '/logos/Crystal_Palace.png' },
  'Everton': { abbreviation: 'EVE', logoPath: '/logos/Everton.png' },
  'Fulham': { abbreviation: 'FUL', logoPath: '/logos/Fulham.png' },
  'Ipswich Town': { abbreviation: 'IPS', logoPath: '/logos/Ipswich_Town.png' },
  'Ipswich': { abbreviation: 'IPS', logoPath: '/logos/Ipswich_Town.png' }, // graph / FPL short name
  'Liverpool': { abbreviation: 'LIV', logoPath: '/logos/Liverpool.png' },
  'Manchester City': { abbreviation: 'MCI', logoPath: '/logos/Man_City.png' },
  'Man City': { abbreviation: 'MCI', logoPath: '/logos/Man_City.png' },
  'Manchester United': { abbreviation: 'MUN', logoPath: '/logos/Man_United.png' },
  'Man Utd': { abbreviation: 'MUN', logoPath: '/logos/Man_United.png' },
  'Man United': { abbreviation: 'MUN', logoPath: '/logos/Man_United.png' },
  'Newcastle': { abbreviation: 'NEW', logoPath: '/logos/Newcastle.png' },
  'Newcastle United': { abbreviation: 'NEW', logoPath: '/logos/Newcastle.png' },
  'Nottingham Forest': { abbreviation: 'NFO', logoPath: '/logos/Nottingham_Forest.png' },
  "Nott'm Forest": { abbreviation: 'NFO', logoPath: '/logos/Nottingham_Forest.png' },
  'Tottenham': { abbreviation: 'TOT', logoPath: '/logos/Tottenham.png' },
  'Tottenham Hotspur': { abbreviation: 'TOT', logoPath: '/logos/Tottenham.png' },
  'Spurs': { abbreviation: 'TOT', logoPath: '/logos/Tottenham.png' },
  // Championship/Promoted teams that might not be in Premier League
  'Coventry City': { abbreviation: 'COV', logoPath: '/logos/Conventry_City.png' },
  'Hull City': { abbreviation: 'HUL', logoPath: '/logos/Hull_City.png' },
  'Leeds': { abbreviation: 'LEE', logoPath: '/logos/Leeds.png' },
  'Leeds United': { abbreviation: 'LEE', logoPath: '/logos/Leeds.png' },
  'Sunderland': { abbreviation: 'SUN', logoPath: '/logos/Sunderland.png' },
};

/**
 * Get club information (abbreviation and logo path) for a team name
 */
export function getClubInfo(teamName: string | null | undefined): ClubInfo | null {
  if (!teamName) return null;
  
  // Direct lookup
  if (CLUB_MAPPING[teamName]) {
    return CLUB_MAPPING[teamName];
  }
  
  // Case-insensitive lookup
  const normalizedName = teamName.toLowerCase();
  const matchedKey = Object.keys(CLUB_MAPPING).find(
    key => key.toLowerCase() === normalizedName
  );
  
  if (matchedKey) {
    return CLUB_MAPPING[matchedKey];
  }
  
  return null;
}

/**
 * Get club abbreviation for a team name
 */
export function getClubAbbreviation(teamName: string | null | undefined): string {
  const clubInfo = getClubInfo(teamName);
  return clubInfo?.abbreviation || teamName?.substring(0, 3).toUpperCase() || '???';
}

/**
 * Get club logo path for a team name
 */
export function getClubLogoPath(teamName: string | null | undefined): string | null {
  const clubInfo = getClubInfo(teamName);
  return clubInfo?.logoPath || null;
}
