"""
External API Clients
====================
Two clients with shared design principles:
  - Read API keys from config.py (or skip entirely if missing)
  - Cache responses to disk to avoid burning rate-limited free tiers
  - Never crash the predictor — return None on any failure
  - Log clearly so you can see what's working and what isn't

Both clients are designed so the predictor can run with NO keys, ONE key, or
BOTH keys, and gain features progressively as more data becomes available.
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

# Try to import config — fall back to template if user hasn't created it yet
try:
    import config
    ODDS_API_KEY = getattr(config, "ODDS_API_KEY", None)
    API_FOOTBALL_KEY = getattr(config, "API_FOOTBALL_KEY", None)
    CACHE_DIR = Path(getattr(config, "CACHE_DIR", "/home/claude/pl_data/api_cache"))
    ODDS_CACHE_MINUTES = getattr(config, "ODDS_CACHE_MINUTES", 360)
    LINEUP_CACHE_MINUTES = getattr(config, "LINEUP_CACHE_MINUTES", 60)
    PREFERRED_BOOKMAKERS = getattr(config, "PREFERRED_BOOKMAKERS",
                                    ["pinnacle", "bet365", "williamhill"])
except ImportError:
    print("[api_clients] No config.py found — running without external APIs.")
    print("[api_clients]   Copy config_template.py -> config.py and add your keys to enable.")
    ODDS_API_KEY = None
    API_FOOTBALL_KEY = None
    CACHE_DIR = Path("/home/claude/pl_data/api_cache")
    ODDS_CACHE_MINUTES = 360
    LINEUP_CACHE_MINUTES = 60
    PREFERRED_BOOKMAKERS = ["pinnacle", "bet365", "williamhill"]


CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Cache helpers
# =============================================================================
def _cache_path(name):
    return CACHE_DIR / f"{name}.json"


def _read_cache(name, max_age_minutes):
    """Return cached payload if it exists and is fresh enough."""
    path = _cache_path(name)
    if not path.exists():
        return None
    age_min = (time.time() - path.stat().st_mtime) / 60
    if age_min > max_age_minutes:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(name, payload):
    try:
        with open(_cache_path(name), "w") as f:
            json.dump(payload, f)
    except OSError as e:
        print(f"[cache] Couldn't write {name}: {e}")


# =============================================================================
# The Odds API — live pre-match odds
# =============================================================================
class OddsAPIClient:
    """Wraps https://the-odds-api.com/.

    Free tier: 500 requests/month. One call returns ALL upcoming PL fixtures
    for a given market, so you only need ~3 calls per gameweek (1X2, O/U 2.5, BTTS).
    """

    BASE = "https://api.the-odds-api.com/v4"
    SPORT = "soccer_epl"  # English Premier League

    MARKETS = {
        "h2h": "1X2 (home/draw/away) — main market",
        "totals": "Over/Under 2.5 goals",
        "btts": "Both Teams to Score",
    }

    def __init__(self, api_key=None):
        self.key = api_key or ODDS_API_KEY
        self.enabled = self.key is not None and self.key != ""
        self.requests_remaining = None  # populated from response headers
        if not self.enabled:
            print("[odds_api] Disabled — no key in config.py. CSV odds will be used as fallback.")

    def _get(self, endpoint, params=None, cache_name=None, cache_minutes=None):
        if not self.enabled:
            return None

        # try cache first
        if cache_name:
            cached = _read_cache(cache_name, cache_minutes or ODDS_CACHE_MINUTES)
            if cached is not None:
                return cached

        params = dict(params or {})
        params["apiKey"] = self.key

        try:
            r = requests.get(f"{self.BASE}{endpoint}", params=params, timeout=15)
            # surface remaining quota — header set by the API
            self.requests_remaining = r.headers.get("x-requests-remaining")
            if r.status_code == 401:
                print("[odds_api] 401 Unauthorized — your API key looks invalid.")
                return None
            if r.status_code == 429:
                print("[odds_api] 429 — monthly quota exhausted. Falling back to CSV odds.")
                return None
            r.raise_for_status()
            payload = r.json()
            if cache_name:
                _write_cache(cache_name, payload)
            return payload
        except requests.RequestException as e:
            print(f"[odds_api] Request failed: {e}")
            return None

    def get_fixtures_with_odds(self, market="h2h"):
        """Return upcoming PL fixtures with odds for the given market.

        market: 'h2h' for 1X2, 'totals' for over/under, 'btts' for both teams to score.
        """
        if not self.enabled:
            return []

        params = {
            "regions": "uk,eu",
            "markets": market,
            "oddsFormat": "decimal",
        }
        cache_name = f"odds_{market}"
        data = self._get(f"/sports/{self.SPORT}/odds", params=params,
                         cache_name=cache_name, cache_minutes=ODDS_CACHE_MINUTES)
        if data is None:
            return []

        if self.requests_remaining is not None:
            print(f"[odds_api] Fetched {market} odds for {len(data)} fixtures. "
                  f"Quota remaining: {self.requests_remaining}")
        return data

    @staticmethod
    def _best_book_h2h(fixture):
        """Pick the sharpest available bookmaker's h2h prices.
        Returns (home_odd, draw_odd, away_odd) or None.
        """
        for preferred in PREFERRED_BOOKMAKERS:
            for book in fixture.get("bookmakers", []):
                if book["key"] != preferred:
                    continue
                for market in book.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    home, away = fixture["home_team"], fixture["away_team"]
                    if home in outcomes and away in outcomes and "Draw" in outcomes:
                        return outcomes[home], outcomes["Draw"], outcomes[away]
        return None

    @staticmethod
    def _best_book_market(fixture, market_key, outcome_names):
        """Generic helper for non-h2h markets. Returns dict of outcome_name -> odd."""
        for preferred in PREFERRED_BOOKMAKERS:
            for book in fixture.get("bookmakers", []):
                if book["key"] != preferred:
                    continue
                for market in book.get("markets", []):
                    if market["key"] != market_key:
                        continue
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])
                                if o.get("price")}
                    if all(name in outcomes for name in outcome_names):
                        return outcomes
        return None

    def implied_probabilities(self, home_team, away_team):
        """Return implied 1X2 probabilities + over/under + BTTS for one fixture.
        Returns a dict (with None values where unavailable) so callers can pick what they need.
        Result keys: p_home, p_draw, p_away, p_over_25, p_btts_yes
        """
        result = {
            "p_home": None, "p_draw": None, "p_away": None,
            "p_over_25": None, "p_btts_yes": None,
            "source": "odds_api" if self.enabled else None,
        }
        if not self.enabled:
            return result

        # h2h
        h2h_data = self.get_fixtures_with_odds("h2h")
        fixture = self._find_fixture(h2h_data, home_team, away_team)
        if fixture:
            odds = self._best_book_h2h(fixture)
            if odds:
                result["p_home"], result["p_draw"], result["p_away"] = self._normalise(*odds)

        # over/under 2.5 — requires a separate call but cached
        totals_data = self.get_fixtures_with_odds("totals")
        fixture = self._find_fixture(totals_data, home_team, away_team)
        if fixture:
            outcomes = self._best_book_market(fixture, "totals", ["Over", "Under"])
            if outcomes:
                # find 2.5 line specifically — Odds API may return multiple lines
                for book in fixture.get("bookmakers", []):
                    for market in book.get("markets", []):
                        if market["key"] != "totals":
                            continue
                        for o in market.get("outcomes", []):
                            if o.get("point") == 2.5 and o["name"] == "Over":
                                # find paired Under at same line
                                under = next((u for u in market["outcomes"]
                                              if u.get("point") == 2.5 and u["name"] == "Under"), None)
                                if under:
                                    p_over, _ = self._normalise(o["price"], under["price"])
                                    result["p_over_25"] = p_over
                                    break

        # btts
        btts_data = self.get_fixtures_with_odds("btts")
        fixture = self._find_fixture(btts_data, home_team, away_team)
        if fixture:
            outcomes = self._best_book_market(fixture, "btts", ["Yes", "No"])
            if outcomes:
                p_yes, _ = self._normalise(outcomes["Yes"], outcomes["No"])
                result["p_btts_yes"] = p_yes

        return result

    @staticmethod
    def _find_fixture(fixtures, home, away):
        """Match by team names. The Odds API uses full names; CSV uses shortened forms."""
        if not fixtures:
            return None
        # try exact match first
        for fx in fixtures:
            if fx["home_team"] == home and fx["away_team"] == away:
                return fx
        # try fuzzy: any fixture where both teams appear in the names
        for fx in fixtures:
            h_match = home in fx["home_team"] or fx["home_team"] in home
            a_match = away in fx["away_team"] or fx["away_team"] in away
            if h_match and a_match:
                return fx
        return None

    @staticmethod
    def _normalise(*odds):
        """Strip overround. (1/odd1 + 1/odd2 + ...) > 1 by the vig; divide each by total."""
        try:
            inv = [1 / float(o) for o in odds]
            total = sum(inv)
            return tuple(p / total for p in inv)
        except (ValueError, ZeroDivisionError, TypeError):
            return tuple(None for _ in odds)


# =============================================================================
# API-Football — lineups, injuries, suspensions
# =============================================================================
class APIFootballClient:
    """Wraps https://www.api-football.com/.

    Free tier: 100 requests/day. We use it sparingly:
      - One call to get next gameweek's fixtures
      - One call per fixture for predicted lineups (only when within 24h of kickoff)
      - One call for current injuries (cached for the day)
    Total per gameweek: ~12 requests if used carefully.
    """

    BASE = "https://v3.football.api-sports.io"
    PL_LEAGUE_ID = 39  # Premier League
    HEADERS_TEMPLATE = {"x-apisports-key": ""}

    # Players whose absence materially shifts a team's win probability.
    # In practice: top 3-5 players by ICT/expected goal involvements per side.
    # We don't hardcode — we infer "key" from FPL ownership/form and team_id mapping.
    KEY_PLAYER_FORM_THRESHOLD = 6.0  # FPL form score above which a player is "key"

    def __init__(self, api_key=None):
        self.key = api_key or API_FOOTBALL_KEY
        self.enabled = self.key is not None and self.key != ""
        self.requests_today = 0
        if not self.enabled:
            print("[api_football] Disabled — no key in config.py. Lineup/injury features will be skipped.")

    def _headers(self):
        return {"x-apisports-key": self.key}

    def _get(self, endpoint, params=None, cache_name=None, cache_minutes=None):
        if not self.enabled:
            return None

        if cache_name:
            cached = _read_cache(cache_name, cache_minutes or LINEUP_CACHE_MINUTES)
            if cached is not None:
                return cached

        try:
            r = requests.get(f"{self.BASE}{endpoint}", params=params or {},
                             headers=self._headers(), timeout=15)
            self.requests_today += 1
            if r.status_code == 401 or r.status_code == 403:
                print(f"[api_football] {r.status_code} — API key invalid or quota exhausted.")
                return None
            if r.status_code == 429:
                print("[api_football] 429 — daily quota exhausted.")
                return None
            r.raise_for_status()
            payload = r.json()
            # API-Football wraps results in {"response": [...], "errors": [...]}
            if payload.get("errors"):
                print(f"[api_football] API errors: {payload['errors']}")
            if cache_name:
                _write_cache(cache_name, payload)
            return payload
        except requests.RequestException as e:
            print(f"[api_football] Request failed: {e}")
            return None

    def get_injuries(self, season=None):
        """Get all current Premier League injuries.

        Returns: list of {player_name, team_name, type, reason}
        """
        if not self.enabled:
            return []

        if season is None:
            # Football seasons span two calendar years — use the start year.
            now = datetime.now()
            season = now.year if now.month >= 7 else now.year - 1

        data = self._get("/injuries",
                         params={"league": self.PL_LEAGUE_ID, "season": season},
                         cache_name=f"injuries_{season}",
                         cache_minutes=LINEUP_CACHE_MINUTES * 4)  # injuries change slowly
        if not data or not data.get("response"):
            return []

        out = []
        for entry in data["response"]:
            out.append({
                "player_name": entry.get("player", {}).get("name", ""),
                "player_id": entry.get("player", {}).get("id"),
                "team_name": entry.get("team", {}).get("name", ""),
                "team_id": entry.get("team", {}).get("id"),
                "type": entry.get("player", {}).get("type", ""),
                "reason": entry.get("player", {}).get("reason", ""),
            })
        return out

    def get_predicted_lineup(self, fixture_id):
        """Get predicted starting XI for a fixture. Available ~1 hour before kickoff."""
        if not self.enabled:
            return None
        data = self._get("/fixtures/predictions",
                         params={"fixture": fixture_id},
                         cache_name=f"lineup_{fixture_id}",
                         cache_minutes=LINEUP_CACHE_MINUTES)
        if not data or not data.get("response"):
            return None
        return data["response"]

    def count_key_players_out(self, team_name, injuries, key_player_names=None):
        """Count how many of a team's key players are currently injured/suspended.

        key_player_names: list of player names you've flagged as key (from FPL form data).
                          If None, just count any player marked unavailable.
        """
        team_injuries = [
            inj for inj in injuries
            if team_name.lower() in inj["team_name"].lower()
            or inj["team_name"].lower() in team_name.lower()
        ]

        if key_player_names is None:
            return len(team_injuries)

        key_out = sum(
            1 for inj in team_injuries
            if any(kp.lower() in inj["player_name"].lower()
                   or inj["player_name"].lower() in kp.lower()
                   for kp in key_player_names)
        )
        return key_out


# =============================================================================
# Combined enrichment — what the predictor calls
# =============================================================================
def enrich_fixture_features(home_team, away_team, odds_client=None, footy_client=None,
                              key_players_by_team=None, csv_market_probs=None):
    """One-stop function: take a fixture, enrich with whatever external data we can get.

    Returns a dict of features. Missing values are filled with sensible neutrals
    so the model never sees NaN.

    Parameters
    ----------
    home_team, away_team : str
        Team names (as appearing in football-data.co.uk CSV)
    odds_client : OddsAPIClient
    footy_client : APIFootballClient
    key_players_by_team : dict[str, list[str]]
        e.g. {"Arsenal": ["Saka", "Saliba", "Odegaard"], ...} — used to score injury impact
    csv_market_probs : dict
        Fallback probabilities from football-data.co.uk CSV: {"mkt_p_home": ..., ...}
    """
    out = {
        # market — Odds API preferred, CSV fallback
        "mkt_p_home": None, "mkt_p_draw": None, "mkt_p_away": None,
        "mkt_p_over_25": None, "mkt_p_btts_yes": None,
        "mkt_source": "none",
        # injuries
        "h_key_players_out": 0, "a_key_players_out": 0,
        "h_injuries_total": 0, "a_injuries_total": 0,
    }

    # --- odds enrichment ------------------------------------------------------
    if odds_client is not None and odds_client.enabled:
        live = odds_client.implied_probabilities(home_team, away_team)
        if live.get("p_home") is not None:
            out["mkt_p_home"] = live["p_home"]
            out["mkt_p_draw"] = live["p_draw"]
            out["mkt_p_away"] = live["p_away"]
            out["mkt_source"] = "odds_api"
        out["mkt_p_over_25"] = live.get("p_over_25")
        out["mkt_p_btts_yes"] = live.get("p_btts_yes")

    # fall back to CSV odds if Odds API didn't give us 1X2
    if out["mkt_p_home"] is None and csv_market_probs:
        out["mkt_p_home"] = csv_market_probs.get("mkt_p_home")
        out["mkt_p_draw"] = csv_market_probs.get("mkt_p_draw")
        out["mkt_p_away"] = csv_market_probs.get("mkt_p_away")
        out["mkt_source"] = "csv"

    # final fallback — neutral PL long-run averages
    if out["mkt_p_home"] is None:
        out["mkt_p_home"], out["mkt_p_draw"], out["mkt_p_away"] = 0.45, 0.27, 0.28
        out["mkt_source"] = "neutral_default"

    # for over/under and btts, use long-run PL averages if missing
    # (PL avg ~55% over 2.5, ~52% BTTS)
    if out["mkt_p_over_25"] is None:
        out["mkt_p_over_25"] = 0.55
    if out["mkt_p_btts_yes"] is None:
        out["mkt_p_btts_yes"] = 0.52

    # --- injury enrichment ----------------------------------------------------
    if footy_client is not None and footy_client.enabled:
        injuries = footy_client.get_injuries()
        if injuries:
            home_keys = (key_players_by_team or {}).get(home_team)
            away_keys = (key_players_by_team or {}).get(away_team)
            out["h_key_players_out"] = footy_client.count_key_players_out(
                home_team, injuries, home_keys)
            out["a_key_players_out"] = footy_client.count_key_players_out(
                away_team, injuries, away_keys)
            out["h_injuries_total"] = footy_client.count_key_players_out(home_team, injuries)
            out["a_injuries_total"] = footy_client.count_key_players_out(away_team, injuries)

    return out


# =============================================================================
# CLI smoke test — run `python3 api_clients.py` to see what's working
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  API Clients — connectivity test")
    print("=" * 60)

    odds = OddsAPIClient()
    if odds.enabled:
        fx = odds.get_fixtures_with_odds("h2h")
        print(f"\nOdds API: {len(fx)} fixtures returned")
        if fx:
            print(f"  First fixture: {fx[0]['home_team']} vs {fx[0]['away_team']}")
            probs = odds.implied_probabilities(fx[0]["home_team"], fx[0]["away_team"])
            print(f"  Implied probs: {probs}")
    else:
        print("\nOdds API: disabled (no key)")

    footy = APIFootballClient()
    if footy.enabled:
        injuries = footy.get_injuries()
        print(f"\nAPI-Football: {len(injuries)} current PL injuries")
        if injuries:
            print(f"  Sample: {injuries[0]}")
    else:
        print("\nAPI-Football: disabled (no key)")

    print("\nBoth clients are designed to fail safely. The predictor will run")
    print("with whatever data is available.")
