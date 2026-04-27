"""
API Configuration
=================
Copy this file to `config.py` and add your keys.
Add `config.py` to .gitignore so your keys don't end up on GitHub.

Sign up here:
  - The Odds API:  https://the-odds-api.com/         (500 requests/month free)
  - API-Football:  https://www.api-football.com/     (100 requests/day free)
"""

# Paste your API key as a string. Leave as None to disable that integration.
ODDS_API_KEY = None      # e.g. "abc123def456..."
API_FOOTBALL_KEY = None  # e.g. "xyz789..."

# Cache directory — responses cached for 6 hours to save API calls
CACHE_DIR = "/home/claude/pl_data/api_cache"

# How fresh odds need to be (minutes). Below this, hit cache. Above, refetch.
ODDS_CACHE_MINUTES = 360   # 6 hours
LINEUP_CACHE_MINUTES = 60  # 1 hour — lineups change closer to kickoff

# Bookmakers to use from The Odds API. Pinnacle is sharpest. Bet365 most-quoted.
PREFERRED_BOOKMAKERS = ["pinnacle", "bet365", "williamhill", "betfair"]
