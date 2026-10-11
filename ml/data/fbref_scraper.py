"""
FBref Championship Data Scraper

Scrapes player and team statistics from FBref for Championship seasons.
Implements STRICT rate limiting (10 req/min) and TLS fingerprint spoofing
to comply with Sports Reference / Cloudflare protection.

CRITICAL Rate Limits:
- Sports Reference: MAX 10 requests per 60 seconds
- Violation = 24-hour IP ban
- Implementation: 7-9 second delays between requests
"""
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import random

# Use curl_cffi to spoof Chrome's TLS fingerprint and bypass Cloudflare
try:
    from curl_cffi import requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    import requests
    CURL_CFFI_AVAILABLE = False
    logging.warning("curl_cffi not available - FBref may block requests")

logger = logging.getLogger(__name__)


class FBrefScraper:
    """
    Scraper for FBref Championship data.
    
    CRITICAL: Sports Reference enforces 10 requests per minute MAX.
    Exceeding this will result in 24-hour IP ban.
    
    Handles:
    - Strict rate limiting (7-9s between requests)
    - TLS fingerprint spoofing (curl_cffi)
    - Cloudflare bypass headers
    - Player per-90 stats (multiple categories)
    - Team aggregate stats
    """
    
    BASE_URL = "https://fbref.com"
    CHAMPIONSHIP_COMP_ID = "10"  # Championship competition ID on FBref
    
    # CRITICAL: Sports Reference = 10 req/min MAX
    MIN_DELAY = 7.0  # 7 seconds = safe (8.5 req/min)
    MAX_DELAY = 9.0  # Random variance to mimic human
    MAX_RETRIES = 2  # Reduced retries (each retry costs time)
    
    # Request tracking for rate limit enforcement
    REQUEST_WINDOW = 60.0  # 60 seconds
    MAX_REQUESTS_PER_WINDOW = 9  # Stay under 10 to be safe
    
    # Stat categories to scrape
    STAT_CATEGORIES = [
        'stats',       # Standard stats (goals, assists, minutes)
        'shooting',    # Shots, xG
        'passing',     # Key passes, pass completion
        'defense',     # Tackles, interceptions
        'possession',  # Dribbles, progressive carries
        'gca',         # Goal/shot-creating actions
        'playingtime', # Starts, minutes breakdown
    ]
    
    KEEPER_CATEGORIES = [
        'keepers',     # Saves, clean sheets
        'keepersadv',  # Advanced keeper stats (PSxG)
    ]
    
    def __init__(self, rate_limit_delay: float = 7.5):
        """
        Initialize scraper with STRICT rate limiting.
        
        Args:
            rate_limit_delay: Seconds between requests (default 7.5s = safe)
        """
        self.rate_limit_delay = max(rate_limit_delay, self.MIN_DELAY)
        self.last_request_time = 0
        self.request_times = []  # Track all requests in rolling window
        
        # Session with TLS fingerprint spoofing if available
        if CURL_CFFI_AVAILABLE:
            self.session = requests.Session()
            self.impersonate = "chrome120"
            logger.info("Using curl_cffi with Chrome 120 TLS fingerprint")
        else:
            import requests as standard_requests
            self.session = standard_requests.Session()
            self.impersonate = None
            logger.warning("Using standard requests - may be blocked by Cloudflare")
        
        # Full browser headers - CRITICAL for Cloudflare bypass
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            # Chrome Client Hints
            'sec-ch-ua': '"Not A Brand";v="99", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        })
    
    def _enforce_rate_limit(self):
        """
        STRICT rate limit enforcement for Sports Reference.
        
        Rules:
        1. Never exceed 10 requests in any 60-second window
        2. Minimum 7 seconds between consecutive requests
        3. Add random variance to mimic human behavior
        """
        current_time = time.time()
        
        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times 
                             if current_time - t < self.REQUEST_WINDOW]
        
        # If we've hit the limit, wait until oldest request expires
        if len(self.request_times) >= self.MAX_REQUESTS_PER_WINDOW:
            oldest_request = min(self.request_times)
            wait_time = self.REQUEST_WINDOW - (current_time - oldest_request) + 1
            logger.warning(f"Rate limit: {len(self.request_times)} requests in window, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            current_time = time.time()
            self.request_times = [t for t in self.request_times 
                                 if current_time - t < self.REQUEST_WINDOW]
        
        # Enforce minimum delay between requests
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - elapsed + random.uniform(0, 2)
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        # Record this request
        self.request_times.append(time.time())
        self.last_request_time = time.time()
    
    def _fetch_url(self, url: str, retry_count: int = 0) -> Optional[str]:
        """
        Fetch URL with STRICT rate limiting and TLS spoofing.
        
        Args:
            url: URL to fetch
            retry_count: Current retry attempt
            
        Returns:
            HTML content or None on failure
        """
        self._enforce_rate_limit()
        
        try:
            logger.info(f"Fetching: {url}")
            
            # Use curl_cffi to spoof Chrome's TLS fingerprint
            if self.impersonate:
                response = self.session.get(url, timeout=30, impersonate=self.impersonate)
            else:
                response = self.session.get(url, timeout=30)
            
            response.raise_for_status()
            logger.info(f"✓ Success ({response.status_code})")
            return response.text
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            
            if retry_count < self.MAX_RETRIES:
                # Long backoff for 403s - likely Cloudflare challenge
                wait_time = 10 * (2 ** retry_count) + random.uniform(2, 5)
                logger.info(f"Retrying in {wait_time:.1f}s (attempt {retry_count + 1}/{self.MAX_RETRIES})")
                time.sleep(wait_time)
                return self._fetch_url(url, retry_count + 1)
            
            logger.error(f"Max retries exceeded for {url}")
            return None
    
    def scrape_player_stats(self, 
                           season: str,
                           category: str = 'stats') -> Optional[pd.DataFrame]:
        """
        Scrape player statistics for a Championship season.
        
        Args:
            season: Season string (e.g., "2024-2025")
            category: Stat category (e.g., 'stats', 'shooting', 'passing')
            
        Returns:
            DataFrame with player stats or None on failure
        """
        # Construct URL
        # Format: https://fbref.com/en/comps/10/2024-2025/stats/2024-2025-Championship-Stats
        url = f"{self.BASE_URL}/en/comps/{self.CHAMPIONSHIP_COMP_ID}/{season}/{category}/{season}-Championship-Stats"
        
        html = self._fetch_url(url)
        if html is None:
            return None
        
        try:
            # Parse tables using pandas
            tables = pd.read_html(html)
            
            if not tables:
                logger.warning(f"No tables found for {season} {category}")
                return None
            
            # First table is usually player stats
            df = tables[0]
            
            # Clean multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
            
            # Add metadata
            df['season'] = season
            df['stat_category'] = category
            
            logger.info(f"Scraped {len(df)} players for {season} {category}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to parse table for {season} {category}: {e}")
            return None
    
    def scrape_team_stats(self, season: str) -> Optional[pd.DataFrame]:
        """
        Scrape team aggregate statistics for a Championship season.
        
        Args:
            season: Season string (e.g., "2024-2025")
            
        Returns:
            DataFrame with team stats or None on failure
        """
        url = f"{self.BASE_URL}/en/comps/{self.CHAMPIONSHIP_COMP_ID}/{season}/stats/{season}-Championship-Stats"
        
        html = self._fetch_url(url)
        if html is None:
            return None
        
        try:
            # Parse tables
            tables = pd.read_html(html)
            
            # Look for squad stats table (usually first or second table)
            # Team stats typically have "Squad" in the header
            for i, table in enumerate(tables):
                if 'Squad' in str(table.columns):
                    df = table
                    break
            else:
                # Fallback: use first table
                logger.warning(f"Could not identify squad table for {season}, using first table")
                df = tables[0]
            
            # Clean columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
            
            df['season'] = season
            
            logger.info(f"Scraped {len(df)} teams for {season}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to parse team stats for {season}: {e}")
            return None
    
    def scrape_season_all_categories(self, season: str) -> Dict[str, pd.DataFrame]:
        """
        Scrape all stat categories for a season (comprehensive).
        
        Args:
            season: Season string (e.g., "2024-2025")
            
        Returns:
            Dictionary mapping category names to DataFrames
        """
        results = {}
        
        # Scrape outfield player categories
        for category in self.STAT_CATEGORIES:
            logger.info(f"Scraping {category} for {season}...")
            df = self.scrape_player_stats(season, category)
            if df is not None:
                results[category] = df
            time.sleep(1)  # Extra buffer between categories
        
        # Scrape goalkeeper categories
        for category in self.KEEPER_CATEGORIES:
            logger.info(f"Scraping {category} for {season}...")
            df = self.scrape_player_stats(season, category)
            if df is not None:
                results[category] = df
            time.sleep(1)
        
        # Scrape team stats
        logger.info(f"Scraping team stats for {season}...")
        team_df = self.scrape_team_stats(season)
        if team_df is not None:
            results['team_stats'] = team_df
        
        logger.info(f"Completed scraping for {season}: {len(results)} categories")
        return results
    
    def get_seasons_list(self, start_year: int = 2019, end_year: int = 2026) -> List[str]:
        """
        Generate list of season strings for scraping.
        
        Args:
            start_year: Starting year (e.g., 2019 for 2019-2020 season)
            end_year: Ending year (e.g., 2026 for 2025-2026 season)
            
        Returns:
            List of season strings (e.g., ["2019-2020", "2020-2021", ...])
        """
        seasons = []
        for year in range(start_year, end_year):
            seasons.append(f"{year}-{year+1}")
        return seasons


def test_scraper():
    """Test function to verify scraper works."""
    logger.info("Testing FBref scraper...")
    
    scraper = FBrefScraper(rate_limit_delay=4.0)
    
    # Test with most recent complete season
    test_season = "2023-2024"
    
    logger.info(f"Testing with season: {test_season}")
    df = scraper.scrape_player_stats(test_season, 'stats')
    
    if df is not None:
        logger.info(f"✅ Scraper test successful!")
        logger.info(f"   Scraped {len(df)} players")
        logger.info(f"   Columns: {list(df.columns[:10])}...")
        return True
    else:
        logger.error("❌ Scraper test failed")
        return False


if __name__ == '__main__':
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_scraper()
