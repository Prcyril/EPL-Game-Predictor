"""
FPL Predictor — Full Gameweek Predictions
==========================================
Combines:
  - Mariana's PL repo: match data pipeline + rolling form features
  - Mariana's F1 repo: gradient boosting for continuous predictions (goals)
  - FPL official API: live player data (form, price, team, position)

Outputs per gameweek:
  - Match outcome probabilities + expected goals
  - Clean sheet probabilities for each team
  - Top captaincy picks (highest expected points)
  - Best defenders/GKs for clean sheets
  - Top expected points by position (GK, DEF, MID, FWD)

Run:
    python3 fpl_predictor.py
"""

import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# =============================================================================
# 1. MATCH-LEVEL MODEL  (adapted from Mariana's PL repo + F1 gradient boosting)
# =============================================================================

class MatchModel:
    """Predicts match outcome (H/D/A) AND expected goals for each side.

    Outcome  -> RandomForestClassifier  (from PL repo)
    Goals    -> GradientBoostingRegressor x2  (idea borrowed from F1 repo,
                where qualifying time -> race time is regression too)
    """

    SEASON_URLS = {
        "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "2025-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    }

    FEATURES = [
        "home_strength", "away_strength",
        "home_form", "away_form",
        "home_goals_for", "away_goals_for",
        "home_goals_against", "away_goals_against",
    ]

    def __init__(self):
        self.outcome_model = RandomForestClassifier(n_estimators=200, random_state=42)
        # Two regressors — one per side. Same idea as F1 predicting race time.
        self.home_goals_model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, random_state=42
        )
        self.away_goals_model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, random_state=42
        )
        self.matches = None

    # --- data loading ---------------------------------------------------------

    def load_data(self):
        print("Downloading Premier League match data...")
        frames = []
        for season, url in self.SEASON_URLS.items():
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                df = pd.read_csv(StringIO(r.text))
                cleaned = self._clean(df, season)
                if len(cleaned) > 0:
                    frames.append(cleaned)
                    print(f"  {season}: {len(cleaned)} matches")
            except Exception as e:
                print(f"  {season}: skipped ({e})")

        if not frames:
            raise RuntimeError("No match data could be downloaded.")
        self.matches = pd.concat(frames, ignore_index=True)
        print(f"Total: {len(self.matches)} matches\n")

    @staticmethod
    def _clean(df, season):
        rows = []
        for _, r in df.iterrows():
            if pd.isna(r.get("FTHG")) or pd.isna(r.get("FTAG")):
                continue
            try:
                h, a = int(r["FTHG"]), int(r["FTAG"])
            except (ValueError, TypeError):
                continue
            result = "H" if h > a else ("A" if a > h else "D")
            rows.append({
                "season": season,
                "date": r.get("Date", ""),
                "home_team": r.get("HomeTeam", ""),
                "away_team": r.get("AwayTeam", ""),
                "home_goals": h,
                "away_goals": a,
                "result": result,
            })
        return pd.DataFrame(rows)

    # --- feature engineering --------------------------------------------------

    def _team_form(self, history, team):
        """Return rolling stats for a team from its recent matches."""
        if len(history) == 0:
            return {"strength": 50, "form": 5, "gf": 1.4, "ga": 1.4}

        pts = gf = ga = 0
        for _, m in history.iterrows():
            is_home = m["home_team"] == team
            scored = m["home_goals"] if is_home else m["away_goals"]
            conceded = m["away_goals"] if is_home else m["home_goals"]
            gf += scored
            ga += conceded
            if scored > conceded:
                pts += 3
            elif scored == conceded:
                pts += 1

        n = len(history)
        return {
            "strength": min(90, max(10, (pts / n) * 20 + 20)),
            "form": pts,                # last-N points (matches PL repo)
            "gf": gf / n,
            "ga": ga / n,
        }

    def _build_features(self):
        print("Building rolling features...")
        df = self.matches.copy().reset_index(drop=True)

        # init columns
        for c in self.FEATURES:
            df[c] = 0.0

        for i, m in df.iterrows():
            h, a = m["home_team"], m["away_team"]
            past = df.iloc[:i]

            h_hist = past[(past["home_team"] == h) | (past["away_team"] == h)].tail(5)
            a_hist = past[(past["home_team"] == a) | (past["away_team"] == a)].tail(5)

            hs = self._team_form(h_hist, h)
            as_ = self._team_form(a_hist, a)

            df.at[i, "home_strength"] = hs["strength"]
            df.at[i, "away_strength"] = as_["strength"]
            df.at[i, "home_form"] = hs["form"]
            df.at[i, "away_form"] = as_["form"]
            df.at[i, "home_goals_for"] = hs["gf"]
            df.at[i, "away_goals_for"] = as_["gf"]
            df.at[i, "home_goals_against"] = hs["ga"]
            df.at[i, "away_goals_against"] = as_["ga"]

        self.matches = df
        return df

    # --- training -------------------------------------------------------------

    def train(self):
        df = self._build_features()
        # drop the first 50 — features are unreliable with no history
        train = df.iloc[50:].reset_index(drop=True)

        X = train[self.FEATURES]
        y_out = train["result"]
        y_hg = train["home_goals"]
        y_ag = train["away_goals"]

        Xt, Xv, yt_out, yv_out = train_test_split(X, y_out, test_size=0.2, random_state=42)
        _, _, yt_hg, yv_hg = train_test_split(X, y_hg, test_size=0.2, random_state=42)
        _, _, yt_ag, yv_ag = train_test_split(X, y_ag, test_size=0.2, random_state=42)

        self.outcome_model.fit(Xt, yt_out)
        self.home_goals_model.fit(Xt, yt_hg)
        self.away_goals_model.fit(Xt, yt_ag)

        out_acc = accuracy_score(yv_out, self.outcome_model.predict(Xv))
        hg_mae = mean_absolute_error(yv_hg, self.home_goals_model.predict(Xv))
        ag_mae = mean_absolute_error(yv_ag, self.away_goals_model.predict(Xv))

        print(f"\nMATCH MODEL PERFORMANCE")
        print(f"  Outcome accuracy : {out_acc:.1%}")
        print(f"  Home goals MAE   : {hg_mae:.2f}")
        print(f"  Away goals MAE   : {ag_mae:.2f}\n")

    # --- prediction -----------------------------------------------------------

    def predict_match(self, home_team, away_team):
        recent = self.matches.tail(200)
        h_hist = recent[(recent["home_team"] == home_team) | (recent["away_team"] == home_team)].tail(5)
        a_hist = recent[(recent["home_team"] == away_team) | (recent["away_team"] == away_team)].tail(5)

        hs = self._team_form(h_hist, home_team)
        as_ = self._team_form(a_hist, away_team)

        feat = pd.DataFrame([{
            "home_strength": hs["strength"], "away_strength": as_["strength"],
            "home_form": hs["form"], "away_form": as_["form"],
            "home_goals_for": hs["gf"], "away_goals_for": as_["gf"],
            "home_goals_against": hs["ga"], "away_goals_against": as_["ga"],
        }])

        probs = dict(zip(self.outcome_model.classes_, self.outcome_model.predict_proba(feat)[0]))
        xg_home = max(0.0, float(self.home_goals_model.predict(feat)[0]))
        xg_away = max(0.0, float(self.away_goals_model.predict(feat)[0]))

        # Clean sheet probability — Poisson approximation: P(opponent scores 0)
        cs_home = float(np.exp(-xg_away))
        cs_away = float(np.exp(-xg_home))

        return {
            "home_team": home_team, "away_team": away_team,
            "p_home": probs.get("H", 0), "p_draw": probs.get("D", 0), "p_away": probs.get("A", 0),
            "xg_home": xg_home, "xg_away": xg_away,
            "cs_home": cs_home, "cs_away": cs_away,
        }


# =============================================================================
# 2. PLAYER-LEVEL FPL DATA  (FPL official API — free, no auth)
# =============================================================================

class FPLData:
    """Wraps the official Fantasy Premier League API."""

    BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
    FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"

    POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    # The football-data.co.uk feed uses different team names than FPL.
    # Map FPL short names -> football-data.co.uk names where they differ.
    TEAM_NAME_MAP = {
        "Spurs": "Tottenham",
        "Man Utd": "Man United",
        "Nott'm Forest": "Nott'm Forest",
        "Newcastle": "Newcastle",
        "Sheffield Utd": "Sheffield United",
        "Wolves": "Wolves",
    }

    def __init__(self):
        print("Fetching FPL data from official API...")
        boot = requests.get(self.BOOTSTRAP, timeout=15).json()
        self.players = pd.DataFrame(boot["elements"])
        self.teams = pd.DataFrame(boot["teams"])
        self.positions = pd.DataFrame(boot["element_types"])
        self.fixtures = pd.DataFrame(requests.get(self.FIXTURES, timeout=15).json())

        # tidy player frame
        self.players["position"] = self.players["element_type"].map(self.POS_MAP)
        self.players["team_name"] = self.players["team"].map(
            dict(zip(self.teams["id"], self.teams["name"]))
        )
        self.players["team_short"] = self.players["team"].map(
            dict(zip(self.teams["id"], self.teams["short_name"]))
        )
        self.players["price"] = self.players["now_cost"] / 10.0
        self.players["form_f"] = pd.to_numeric(self.players["form"], errors="coerce").fillna(0)
        self.players["ppg"] = pd.to_numeric(self.players["points_per_game"], errors="coerce").fillna(0)
        self.players["xg_per90"] = pd.to_numeric(
            self.players.get("expected_goals_per_90", 0), errors="coerce"
        ).fillna(0)
        self.players["xa_per90"] = pd.to_numeric(
            self.players.get("expected_assists_per_90", 0), errors="coerce"
        ).fillna(0)

        print(f"  Players: {len(self.players)}, Teams: {len(self.teams)}, Fixtures: {len(self.fixtures)}\n")

    def map_to_fbref(self, fpl_name):
        return self.TEAM_NAME_MAP.get(fpl_name, fpl_name)

    def upcoming_fixtures(self, n_gw=1):
        """Return next gameweek's fixtures."""
        future = self.fixtures[self.fixtures["finished"] == False].copy()
        if "event" not in future.columns or future.empty:
            return pd.DataFrame()
        next_gw = int(future["event"].dropna().min())
        gw = future[future["event"] == next_gw].copy()

        team_short = dict(zip(self.teams["id"], self.teams["short_name"]))
        team_name = dict(zip(self.teams["id"], self.teams["name"]))
        gw["home_short"] = gw["team_h"].map(team_short)
        gw["away_short"] = gw["team_a"].map(team_short)
        gw["home_name"] = gw["team_h"].map(team_name)
        gw["away_name"] = gw["team_a"].map(team_name)
        gw["gameweek"] = next_gw
        return gw[["gameweek", "home_short", "away_short", "home_name", "away_name", "kickoff_time"]]


# =============================================================================
# 3. PLAYER POINTS MODEL  (combines match xG with player form)
# =============================================================================

class PlayerPoints:
    """Estimates expected FPL points per player for a given match.

    Logic (transparent, not a black box):
      Attackers (MID/FWD)  : base FPL form * fixture-difficulty multiplier
                             + bonus from team xG share via xG/90 & xA/90
      Defenders/GKs        : base form + clean-sheet probability bonus
                             (4 pts if CS for DEF, 4 pts for GK, 1 pt for MID)
    """

    # FPL scoring for clean sheets
    CS_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
    # FPL goal points
    GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
    ASSIST_POINTS = 3
    APPEARANCE = 2  # 60+ mins

    def __init__(self, match_model: MatchModel, fpl: FPLData):
        self.mm = match_model
        self.fpl = fpl

    def _fixture_difficulty_mult(self, p_win, p_draw):
        """Convert win/draw probability into an attacking multiplier.
        Easy fixture (high p_win) -> >1.0, hard fixture -> <1.0."""
        # centred around 0.45 (roughly an even matchup for the attacking side)
        return 0.7 + (p_win + 0.5 * p_draw) * 0.8  # range ~[0.7, 1.5]

    def predict_for_match(self, match_pred, home_short, away_short):
        """Score every player in either of the two teams for this fixture."""
        rows = []
        for side in ("home", "away"):
            short = home_short if side == "home" else away_short
            xg_team = match_pred[f"xg_{side}"]
            cs_prob = match_pred[f"cs_{side}"]
            p_win = match_pred[f"p_{side}"] if side == "home" else match_pred["p_away"]
            p_draw = match_pred["p_draw"]
            mult = self._fixture_difficulty_mult(p_win, p_draw)

            squad = self.fpl.players[self.fpl.players["team_short"] == short]
            for _, p in squad.iterrows():
                # skip very low-minute players
                if p["minutes"] < 90 and p["form_f"] == 0:
                    continue

                pos = p["position"]
                base = p["ppg"] * mult  # per-game baseline scaled by fixture

                # attacking bonus from xG share
                attacking = (p["xg_per90"] * self.GOAL_POINTS[pos] +
                             p["xa_per90"] * self.ASSIST_POINTS) * (xg_team / 1.4)

                # clean sheet bonus
                cs_bonus = self.CS_POINTS[pos] * cs_prob

                expected = base * 0.6 + attacking * 0.3 + cs_bonus + self.APPEARANCE * 0.4

                rows.append({
                    "player": p["web_name"],
                    "team": short,
                    "position": pos,
                    "price": p["price"],
                    "form": p["form_f"],
                    "ppg": p["ppg"],
                    "opponent": away_short if side == "home" else home_short,
                    "venue": "H" if side == "home" else "A",
                    "cs_prob": round(cs_prob, 2),
                    "xg_team": round(xg_team, 2),
                    "expected_points": round(expected, 2),
                })
        return pd.DataFrame(rows)


# =============================================================================
# 4. GAMEWEEK REPORT
# =============================================================================

def run_gameweek_report():
    print("=" * 70)
    print("  FPL GAMEWEEK PREDICTOR")
    print("=" * 70 + "\n")

    # 1. train match model
    mm = MatchModel()
    mm.load_data()
    mm.train()

    # 2. fetch FPL data
    fpl = FPLData()
    fixtures = fpl.upcoming_fixtures()
    if fixtures.empty:
        print("No upcoming fixtures found in FPL API.")
        return

    gw = int(fixtures["gameweek"].iloc[0])
    print(f"=== GAMEWEEK {gw} — {len(fixtures)} fixtures ===\n")

    # 3. predict each match + each player
    pp = PlayerPoints(mm, fpl)
    all_match_preds = []
    all_player_preds = []

    for _, fx in fixtures.iterrows():
        h_fbref = fpl.map_to_fbref(fx["home_name"])
        a_fbref = fpl.map_to_fbref(fx["away_name"])
        try:
            mp = mm.predict_match(h_fbref, a_fbref)
        except Exception as e:
            print(f"  Skipping {fx['home_short']} vs {fx['away_short']}: {e}")
            continue

        all_match_preds.append({
            "match": f"{fx['home_short']} vs {fx['away_short']}",
            "p_home": f"{mp['p_home']:.0%}",
            "p_draw": f"{mp['p_draw']:.0%}",
            "p_away": f"{mp['p_away']:.0%}",
            "xg_home": round(mp["xg_home"], 2),
            "xg_away": round(mp["xg_away"], 2),
            "cs_home": f"{mp['cs_home']:.0%}",
            "cs_away": f"{mp['cs_away']:.0%}",
        })

        players = pp.predict_for_match(mp, fx["home_short"], fx["away_short"])
        all_player_preds.append(players)

    # 4. summarise
    matches_df = pd.DataFrame(all_match_preds)
    print("\n--- MATCH OUTCOMES + xG + CLEAN SHEET ODDS ---")
    print(matches_df.to_string(index=False))

    if not all_player_preds:
        return
    players_df = pd.concat(all_player_preds, ignore_index=True)

    print("\n\n--- TOP 10 CAPTAINCY PICKS (highest expected points) ---")
    cap = players_df.sort_values("expected_points", ascending=False).head(10)
    print(cap[["player", "team", "position", "opponent", "venue",
               "expected_points", "form", "price"]].to_string(index=False))

    print("\n\n--- TOP 5 BY POSITION ---")
    for pos in ("GK", "DEF", "MID", "FWD"):
        top = players_df[players_df["position"] == pos] \
            .sort_values("expected_points", ascending=False).head(5)
        print(f"\n  {pos}:")
        print(top[["player", "team", "opponent", "venue",
                   "cs_prob", "expected_points", "price"]].to_string(index=False))

    print("\n\n--- BEST CLEAN SHEET BETS (DEF / GK only) ---")
    cs = players_df[players_df["position"].isin(["GK", "DEF"])] \
        .drop_duplicates(subset=["team", "position"]) \
        .sort_values("cs_prob", ascending=False).head(8)
    print(cs[["player", "team", "position", "opponent", "venue",
              "cs_prob", "expected_points"]].to_string(index=False))

    # save full output
    out_path = "/mnt/user-data/outputs/fpl_gameweek_predictions.csv"
    players_df.sort_values("expected_points", ascending=False).to_csv(out_path, index=False)
    matches_path = "/mnt/user-data/outputs/fpl_gameweek_matches.csv"
    matches_df.to_csv(matches_path, index=False)
    print(f"\n\nFull player table saved -> {out_path}")
    print(f"Match table saved        -> {matches_path}")


if __name__ == "__main__":
    run_gameweek_report()
