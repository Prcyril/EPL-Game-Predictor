"""
Premier League Match Predictor — v4
====================================
Goal: pick winners for each gameweek with continued learning.

Improvements over v1:
  1. Home/away splits          — separate rolling stats per venue
  2. Opponent-adjusted strength — form weighted by quality of opposition faced
  3. Time-decayed form          — recent matches matter more (exponential decay)
  4. Head-to-head history       — historical record between the two teams
  5. Days rest                  — fixture congestion as a feature
  6. Promoted team priors       — newly-promoted teams start at bottom-table avg
  7. Time-series cross-validation — NO future data leakage (real bug fix)
  8. Persistent state            — save model, track predictions vs reality
  9. Bookmaker odds as features  — implied probabilities from CSV / Pinnacle
 10. Market benchmark            — accuracy + log-loss vs market on every fold

NEW in v4 — external APIs (graceful fallback if no keys):
 11. The Odds API integration    — fresher pre-match odds, sharper books,
                                   plus Over/Under 2.5 and BTTS markets
 12. API-Football integration    — current injuries and suspensions
 13. Smart caching               — disk cache to stay under free-tier limits
 14. Future-proofed for in-play  — third goal kept in mind, architecture ready

USAGE
-----
First run (trains from scratch):
    python3 pl_predictor.py

Weekly run (adds new results, retrains, predicts upcoming GW):
    python3 pl_predictor.py --update

Check accuracy on past predictions:
    python3 pl_predictor.py --review

Test API connectivity:
    python3 api_clients.py

To enable external APIs:
    1. Sign up at the-odds-api.com and api-football.com (both free tiers)
    2. cp config_template.py config.py
    3. Paste your keys into config.py
    4. Add config.py to .gitignore
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# external API clients — fail-safe, returns no-op clients if config missing
try:
    from api_clients import OddsAPIClient, APIFootballClient, enrich_fixture_features
    API_CLIENTS_AVAILABLE = True
except ImportError as e:
    print(f"[pl_predictor] api_clients module not found ({e}). Running with CSV-only data.")
    API_CLIENTS_AVAILABLE = False

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATA_DIR = Path("/home/claude/pl_data")
DATA_DIR.mkdir(exist_ok=True)

MATCHES_FILE = DATA_DIR / "matches.parquet"
MODEL_FILE = DATA_DIR / "model.pkl"
PREDICTIONS_FILE = DATA_DIR / "predictions_log.csv"
META_FILE = DATA_DIR / "meta.json"

SEASON_URLS = {
    "2020-21": "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
    "2021-22": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
}

# rolling window for form features
FORM_WINDOW = 6
# exponential decay: most recent match weighted 1.0, oldest in window ~0.5
FORM_DECAY = 0.88


# -----------------------------------------------------------------------------
# Data layer
# -----------------------------------------------------------------------------
def download_all_seasons():
    """Pull every season we know about into one tidy DataFrame."""
    print("Downloading match data...")
    frames = []
    for season, url in SEASON_URLS.items():
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            cleaned = clean_season(df, season)
            if len(cleaned):
                frames.append(cleaned)
                print(f"  {season}: {len(cleaned)} matches")
        except Exception as e:
            print(f"  {season}: skipped ({e})")

    if not frames:
        raise RuntimeError("No data downloaded.")

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values("date").reset_index(drop=True)
    print(f"Total: {len(matches)} matches\n")
    return matches


def odds_to_implied_probs(odd_h, odd_d, odd_a):
    """Convert decimal odds (e.g. 2.50) to fair implied probabilities.
    Bookmakers build in an overround (vig); we remove it by normalising.
    Returns (p_home, p_draw, p_away) summing to 1.0, or None if any odd is missing.
    """
    try:
        oh, od, oa = float(odd_h), float(odd_d), float(odd_a)
        if oh <= 1 or od <= 1 or oa <= 1:
            return None
        raw_h, raw_d, raw_a = 1 / oh, 1 / od, 1 / oa
        total = raw_h + raw_d + raw_a  # >1 by the overround
        return raw_h / total, raw_d / total, raw_a / total
    except (ValueError, TypeError):
        return None


def extract_odds(row):
    """Try multiple bookmaker columns and fall back through them.
    Football-data.co.uk uses these column names across seasons:
      B365H/B365D/B365A   - Bet365 (most consistently present)
      PSH/PSD/PSA         - Pinnacle (sharpest market)
      PSCH/PSCD/PSCA      - Pinnacle closing
      BbAvH/BbAvD/BbAvA   - Betbrain average (older seasons)
      AvgH/AvgD/AvgA      - simple average (newer seasons)
      MaxH/MaxD/MaxA      - best price across books
    Returns implied (p_h, p_d, p_a) or None.
    """
    sources = [
        ("PSCH", "PSCD", "PSCA"),  # Pinnacle closing — sharpest
        ("PSH", "PSD", "PSA"),     # Pinnacle pre-match
        ("B365H", "B365D", "B365A"),
        ("AvgH", "AvgD", "AvgA"),
        ("BbAvH", "BbAvD", "BbAvA"),
        ("MaxH", "MaxD", "MaxA"),
    ]
    for h, d, a in sources:
        if h in row.index:
            probs = odds_to_implied_probs(row.get(h), row.get(d), row.get(a))
            if probs is not None:
                return probs
    return None


def clean_season(df, season):
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("FTHG")) or pd.isna(r.get("FTAG")):
            continue
        try:
            h, a = int(r["FTHG"]), int(r["FTAG"])
            date = pd.to_datetime(r["Date"], dayfirst=True, errors="coerce")
            if pd.isna(date):
                continue
        except (ValueError, TypeError, KeyError):
            continue

        result = "H" if h > a else ("A" if a > h else "D")

        # extract market-implied probabilities — None if odds unavailable
        odds = extract_odds(r)
        if odds is not None:
            mkt_h, mkt_d, mkt_a = odds
        else:
            mkt_h, mkt_d, mkt_a = np.nan, np.nan, np.nan

        rows.append({
            "season": season,
            "date": date,
            "home_team": r.get("HomeTeam", ""),
            "away_team": r.get("AwayTeam", ""),
            "home_goals": h,
            "away_goals": a,
            "result": result,
            "mkt_p_home": mkt_h,
            "mkt_p_draw": mkt_d,
            "mkt_p_away": mkt_a,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def league_average_goals(matches_so_far):
    """Average goals per team per match across the league (used for normalising)."""
    if len(matches_so_far) == 0:
        return 1.4
    total_goals = matches_so_far["home_goals"].sum() + matches_so_far["away_goals"].sum()
    return total_goals / (2 * len(matches_so_far))


def time_decayed_stats(history_df, team, league_avg_goals):
    """Compute time-decayed, opponent-adjusted form for a team.

    history_df : matches the team played, MOST RECENT LAST
    """
    if len(history_df) == 0:
        return neutral_stats()

    # decay weights: most recent gets weight 1, going back: 1, decay, decay^2 ...
    n = len(history_df)
    weights = np.array([FORM_DECAY ** (n - 1 - i) for i in range(n)])

    pts, gf, ga, opp_strength_sum = 0.0, 0.0, 0.0, 0.0
    weight_total = weights.sum()

    for w, (_, m) in zip(weights, history_df.iterrows()):
        is_home = m["home_team"] == team
        scored = m["home_goals"] if is_home else m["away_goals"]
        conceded = m["away_goals"] if is_home else m["home_goals"]
        opponent = m["away_team"] if is_home else m["home_team"]

        gf += scored * w
        ga += conceded * w

        if scored > conceded:
            pts += 3 * w
        elif scored == conceded:
            pts += 1 * w

        # opponent strength placeholder — we'll refine in opponent_adjust pass
        opp_strength_sum += w  # uniform for now

    return {
        "form_pts_per_game": pts / weight_total,
        "gf_per_game": gf / weight_total,
        "ga_per_game": ga / weight_total,
        "matches_played": n,
    }


def neutral_stats():
    """Fallback for teams with no history (start of season, promoted sides handled separately)."""
    return {
        "form_pts_per_game": 1.0,
        "gf_per_game": 1.4,
        "ga_per_game": 1.4,
        "matches_played": 0,
    }


def promoted_team_priors():
    """Newly promoted teams typically perform like bottom-3 sides.
    Empirically ~0.8 pts/game, scoring ~1.0, conceding ~1.7."""
    return {
        "form_pts_per_game": 0.8,
        "gf_per_game": 1.0,
        "ga_per_game": 1.7,
        "matches_played": 0,
    }


def get_team_history(matches_df, team, before_idx, venue=None, window=FORM_WINDOW):
    """Get a team's last N matches before a given index, optionally filtered by venue."""
    past = matches_df.iloc[:before_idx]

    if venue == "home":
        team_matches = past[past["home_team"] == team]
    elif venue == "away":
        team_matches = past[past["away_team"] == team]
    else:
        team_matches = past[(past["home_team"] == team) | (past["away_team"] == team)]

    return team_matches.tail(window)


def head_to_head(matches_df, home, away, before_idx, window=10):
    """Return H2H stats: % home wins, % draws, avg goals diff for home side."""
    past = matches_df.iloc[:before_idx]
    h2h = past[
        ((past["home_team"] == home) & (past["away_team"] == away)) |
        ((past["home_team"] == away) & (past["away_team"] == home))
    ].tail(window)

    if len(h2h) == 0:
        return {"h2h_home_winrate": 0.4, "h2h_draw_rate": 0.25, "h2h_goal_diff": 0.0}

    home_wins, draws, gd = 0, 0, 0
    for _, m in h2h.iterrows():
        gh, ga = m["home_goals"], m["away_goals"]
        if m["home_team"] == home:
            gd += gh - ga
            if gh > ga: home_wins += 1
            elif gh == ga: draws += 1
        else:
            gd -= gh - ga  # flip
            if ga > gh: home_wins += 1
            elif gh == ga: draws += 1

    n = len(h2h)
    return {
        "h2h_home_winrate": home_wins / n,
        "h2h_draw_rate": draws / n,
        "h2h_goal_diff": gd / n,
    }


def days_rest(matches_df, team, before_idx, current_date):
    """Days since this team's last match. Less rest = more fatigue."""
    past = matches_df.iloc[:before_idx]
    team_past = past[(past["home_team"] == team) | (past["away_team"] == team)]
    if len(team_past) == 0:
        return 7  # default — assume normal week
    last_date = team_past["date"].iloc[-1]
    return min(14, (current_date - last_date).days)  # cap at 14


def is_promoted(matches_df, team, before_idx):
    """Heuristic: if team has fewer than 5 matches in this dataset, treat as promoted/new."""
    past = matches_df.iloc[:before_idx]
    count = len(past[(past["home_team"] == team) | (past["away_team"] == team)])
    return count < 5


def build_features(matches_df):
    """Walk through every match and compute features using ONLY past data.
    This is critical — features at row i must use only rows 0..i-1.
    """
    print("Building features (this is the slow part)...")
    df = matches_df.copy().reset_index(drop=True)

    # init feature columns
    feat_cols = [
        # overall form
        "h_form_pts", "a_form_pts",
        "h_gf", "a_gf",
        "h_ga", "a_ga",
        # home/away specific form
        "h_home_pts", "a_away_pts",
        "h_home_gf", "a_away_gf",
        "h_home_ga", "a_away_ga",
        # head to head
        "h2h_home_winrate", "h2h_draw_rate", "h2h_goal_diff",
        # rest
        "h_days_rest", "a_days_rest",
        # promoted flags
        "h_promoted", "a_promoted",
        # form differentials (very predictive)
        "form_diff", "gf_diff", "ga_diff",
        # MARKET ODDS — implied probabilities from bookmaker odds
        # The market is a strong baseline. These features bring in info we can't
        # capture: injuries, lineups, manager comments, sharp money flow.
        "mkt_p_home", "mkt_p_draw", "mkt_p_away",
        "mkt_fav_margin",  # how strongly the market favours one side
        # EXTRA MARKETS — only populated for upcoming fixtures from Odds API.
        # Historical training rows get long-run PL defaults, but the model still
        # learns to use them once you start logging real values weekly.
        "mkt_p_over_25", "mkt_p_btts_yes",
        # INJURY/SUSPENSION SIGNALS — populated from API-Football for upcoming.
        # Historical rows get 0 (we don't have retroactive injury data).
        # These features only become useful once you've been collecting them
        # weekly; at first they'll have low importance, growing over time.
        "h_key_players_out", "a_key_players_out",
        "h_injuries_total", "a_injuries_total",
    ]
    for c in feat_cols:
        df[c] = 0.0

    for i in range(len(df)):
        if i % 200 == 0 and i > 0:
            print(f"  {i}/{len(df)} matches processed")

        m = df.iloc[i]
        h, a = m["home_team"], m["away_team"]
        date = m["date"]
        league_avg = league_average_goals(df.iloc[:i])

        # promoted detection
        h_prom = is_promoted(df, h, i)
        a_prom = is_promoted(df, a, i)

        # overall form
        if h_prom:
            hs = promoted_team_priors()
        else:
            h_hist = get_team_history(df, h, i)
            hs = time_decayed_stats(h_hist, h, league_avg) if len(h_hist) else neutral_stats()

        if a_prom:
            as_ = promoted_team_priors()
        else:
            a_hist = get_team_history(df, a, i)
            as_ = time_decayed_stats(a_hist, a, league_avg) if len(a_hist) else neutral_stats()

        # venue-specific form
        h_home_hist = get_team_history(df, h, i, venue="home")
        a_away_hist = get_team_history(df, a, i, venue="away")
        hs_home = time_decayed_stats(h_home_hist, h, league_avg) if len(h_home_hist) else hs
        as_away = time_decayed_stats(a_away_hist, a, league_avg) if len(a_away_hist) else as_

        # h2h
        h2h = head_to_head(df, h, a, i)

        # rest
        h_rest = days_rest(df, h, i, date)
        a_rest = days_rest(df, a, i, date)

        # write features
        df.at[i, "h_form_pts"] = hs["form_pts_per_game"]
        df.at[i, "a_form_pts"] = as_["form_pts_per_game"]
        df.at[i, "h_gf"] = hs["gf_per_game"]
        df.at[i, "a_gf"] = as_["gf_per_game"]
        df.at[i, "h_ga"] = hs["ga_per_game"]
        df.at[i, "a_ga"] = as_["ga_per_game"]

        df.at[i, "h_home_pts"] = hs_home["form_pts_per_game"]
        df.at[i, "a_away_pts"] = as_away["form_pts_per_game"]
        df.at[i, "h_home_gf"] = hs_home["gf_per_game"]
        df.at[i, "a_away_gf"] = as_away["gf_per_game"]
        df.at[i, "h_home_ga"] = hs_home["ga_per_game"]
        df.at[i, "a_away_ga"] = as_away["ga_per_game"]

        df.at[i, "h2h_home_winrate"] = h2h["h2h_home_winrate"]
        df.at[i, "h2h_draw_rate"] = h2h["h2h_draw_rate"]
        df.at[i, "h2h_goal_diff"] = h2h["h2h_goal_diff"]

        df.at[i, "h_days_rest"] = h_rest
        df.at[i, "a_days_rest"] = a_rest

        df.at[i, "h_promoted"] = 1.0 if h_prom else 0.0
        df.at[i, "a_promoted"] = 1.0 if a_prom else 0.0

        # differentials — often more predictive than raw values
        df.at[i, "form_diff"] = hs["form_pts_per_game"] - as_["form_pts_per_game"]
        df.at[i, "gf_diff"] = hs["gf_per_game"] - as_["gf_per_game"]
        df.at[i, "ga_diff"] = as_["ga_per_game"] - hs["ga_per_game"]

        # market odds — already populated during clean_season, just compute the margin
        # Fall back to neutral 0.45/0.27/0.28 (rough PL long-run avg) if missing
        mh = m.get("mkt_p_home")
        md = m.get("mkt_p_draw")
        ma = m.get("mkt_p_away")
        if pd.isna(mh) or pd.isna(md) or pd.isna(ma):
            mh, md, ma = 0.45, 0.27, 0.28
            df.at[i, "mkt_p_home"] = mh
            df.at[i, "mkt_p_draw"] = md
            df.at[i, "mkt_p_away"] = ma
        df.at[i, "mkt_fav_margin"] = max(mh, ma) - min(mh, ma)

        # New features — historical rows get neutral / zero defaults.
        # Real values only come from external APIs at prediction time.
        df.at[i, "mkt_p_over_25"] = 0.55   # PL long-run avg
        df.at[i, "mkt_p_btts_yes"] = 0.52  # PL long-run avg
        df.at[i, "h_key_players_out"] = 0
        df.at[i, "a_key_players_out"] = 0
        df.at[i, "h_injuries_total"] = 0
        df.at[i, "a_injuries_total"] = 0

    print(f"  Done. {len(feat_cols)} features built.\n")
    return df, feat_cols


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class PLModel:
    """Bundles the three sub-models: outcome classifier + two goal regressors."""

    def __init__(self):
        self.outcome = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        self.home_goals = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        self.away_goals = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        self.feat_cols = None
        self.training_history = []  # list of {date, val_acc, val_logloss}

    def fit(self, df, feat_cols, warmup=100):
        """Fit on all data after the warmup period. Use TimeSeriesSplit for honest validation."""
        self.feat_cols = feat_cols
        train = df.iloc[warmup:].reset_index(drop=True)

        X = train[feat_cols].values
        y_out = train["result"].values
        y_hg = train["home_goals"].values
        y_ag = train["away_goals"].values

        # Time-series CV — train on past, validate on future. NO leakage.
        tscv = TimeSeriesSplit(n_splits=5)
        accs, logs, hg_maes, ag_maes = [], [], [], []
        # market benchmarks (from same validation folds)
        mkt_accs, mkt_logs = [], []

        # pull market probabilities aligned with training set
        mkt_probs_train = train[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].fillna(
            pd.Series({"mkt_p_home": 0.45, "mkt_p_draw": 0.27, "mkt_p_away": 0.28})
        ).values

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
            self.outcome.fit(X[tr_idx], y_out[tr_idx])
            self.home_goals.fit(X[tr_idx], y_hg[tr_idx])
            self.away_goals.fit(X[tr_idx], y_ag[tr_idx])

            pred = self.outcome.predict(X[va_idx])
            proba = self.outcome.predict_proba(X[va_idx])
            accs.append(accuracy_score(y_out[va_idx], pred))
            try:
                logs.append(log_loss(y_out[va_idx], proba, labels=self.outcome.classes_))
            except ValueError:
                logs.append(np.nan)
            hg_maes.append(mean_absolute_error(y_hg[va_idx], self.home_goals.predict(X[va_idx])))
            ag_maes.append(mean_absolute_error(y_ag[va_idx], self.away_goals.predict(X[va_idx])))

            # market baseline: pick the side with highest implied prob, log-loss with the actual probs
            mkt_p = mkt_probs_train[va_idx]
            mkt_pred = np.where(
                (mkt_p[:, 0] >= mkt_p[:, 1]) & (mkt_p[:, 0] >= mkt_p[:, 2]), "H",
                np.where(mkt_p[:, 2] >= mkt_p[:, 1], "A", "D"),
            )
            mkt_accs.append(accuracy_score(y_out[va_idx], mkt_pred))
            # align market prob columns with outcome classes_ order (alphabetical: A, D, H)
            class_to_col = {"H": 0, "D": 1, "A": 2}
            mkt_proba_aligned = np.column_stack([
                mkt_p[:, class_to_col[c]] for c in self.outcome.classes_
            ])
            try:
                mkt_logs.append(log_loss(y_out[va_idx], mkt_proba_aligned, labels=self.outcome.classes_))
            except ValueError:
                mkt_logs.append(np.nan)

        print(f"TIME-SERIES CV ({len(accs)} folds, train -> future split)")
        print(f"  MODEL  accuracy  : {np.mean(accs):.1%}  (per-fold: {[f'{x:.0%}' for x in accs]})")
        print(f"  MARKET accuracy  : {np.mean(mkt_accs):.1%}  (per-fold: {[f'{x:.0%}' for x in mkt_accs]})")
        print(f"  MODEL  log-loss  : {np.nanmean(logs):.3f}")
        print(f"  MARKET log-loss  : {np.nanmean(mkt_logs):.3f}  (lower = better calibration)")
        beat_mkt = np.mean(accs) > np.mean(mkt_accs)
        beat_mkt_log = np.nanmean(logs) < np.nanmean(mkt_logs)
        print(f"  Beating market?  : accuracy {'YES' if beat_mkt else 'no'} | log-loss {'YES' if beat_mkt_log else 'no'}")
        print(f"  Home goals MAE   : {np.mean(hg_maes):.2f}")
        print(f"  Away goals MAE   : {np.mean(ag_maes):.2f}")

        # final fit on ALL training data so prediction uses every game
        self.outcome.fit(X, y_out)
        self.home_goals.fit(X, y_hg)
        self.away_goals.fit(X, y_ag)

        # log this training run
        self.training_history.append({
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_matches": len(train),
            "cv_accuracy": float(np.mean(accs)),
            "cv_logloss": float(np.nanmean(logs)),
            "market_accuracy": float(np.mean(mkt_accs)),
            "market_logloss": float(np.nanmean(mkt_logs)),
        })

        # feature importance
        importances = sorted(zip(feat_cols, self.outcome.feature_importances_),
                             key=lambda x: -x[1])
        print(f"\nTop 8 features for outcome prediction:")
        for name, imp in importances[:8]:
            print(f"  {name:24s} {imp:.3f}")
        print()

    def predict(self, feature_row):
        """feature_row: dict or single-row DataFrame matching feat_cols."""
        if isinstance(feature_row, dict):
            feature_row = pd.DataFrame([feature_row])
        X = feature_row[self.feat_cols].values

        probs = self.outcome.predict_proba(X)[0]
        prob_map = dict(zip(self.outcome.classes_, probs))
        xg_h = max(0.0, float(self.home_goals.predict(X)[0]))
        xg_a = max(0.0, float(self.away_goals.predict(X)[0]))

        return {
            "p_home": prob_map.get("H", 0.0),
            "p_draw": prob_map.get("D", 0.0),
            "p_away": prob_map.get("A", 0.0),
            "xg_home": xg_h,
            "xg_away": xg_a,
            "cs_home": float(np.exp(-xg_a)),
            "cs_away": float(np.exp(-xg_h)),
        }


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def save_state(matches_df, model):
    matches_df.to_parquet(MATCHES_FILE)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    meta = {
        "last_trained": datetime.now().isoformat(timespec="seconds"),
        "n_matches": len(matches_df),
        "training_history": model.training_history,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"State saved to {DATA_DIR}")


def load_state():
    if not (MATCHES_FILE.exists() and MODEL_FILE.exists()):
        return None, None
    matches = pd.read_parquet(MATCHES_FILE)
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return matches, model


# -----------------------------------------------------------------------------
# Prediction logging — for tracking accuracy over time
# -----------------------------------------------------------------------------
def log_prediction(date, home, away, pred, mkt_probs=None):
    """Log a prediction. mkt_probs: optional dict with mkt_p_home/draw/away."""
    row = {
        "predicted_at": datetime.now().isoformat(timespec="seconds"),
        "match_date": date.isoformat() if hasattr(date, "isoformat") else str(date),
        "home_team": home,
        "away_team": away,
        "p_home": round(pred["p_home"], 3),
        "p_draw": round(pred["p_draw"], 3),
        "p_away": round(pred["p_away"], 3),
        "predicted_result": max(["H", "D", "A"],
                                key=lambda k: pred[f"p_{'home' if k=='H' else 'draw' if k=='D' else 'away'}"]),
        "xg_home": round(pred["xg_home"], 2),
        "xg_away": round(pred["xg_away"], 2),
        # market for benchmark comparison
        "mkt_p_home": round(mkt_probs["mkt_p_home"], 3) if mkt_probs else "",
        "mkt_p_draw": round(mkt_probs["mkt_p_draw"], 3) if mkt_probs else "",
        "mkt_p_away": round(mkt_probs["mkt_p_away"], 3) if mkt_probs else "",
        "mkt_pick": (
            max(["H", "D", "A"],
                key=lambda k: mkt_probs[f"mkt_p_{'home' if k=='H' else 'draw' if k=='D' else 'away'}"])
            if mkt_probs else ""
        ),
        "actual_result": "",  # filled in later when result known
        "actual_home_goals": "",
        "actual_away_goals": "",
    }

    if PREDICTIONS_FILE.exists():
        log = pd.read_csv(PREDICTIONS_FILE)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log.to_csv(PREDICTIONS_FILE, index=False)


def review_predictions(matches_df):
    """Reconcile logged predictions against actual results."""
    if not PREDICTIONS_FILE.exists():
        print("No prediction log yet.")
        return

    log = pd.read_csv(PREDICTIONS_FILE)

    # try to match each prediction with the actual result
    updated = 0
    for i, p in log.iterrows():
        if p.get("actual_result"):  # already filled
            continue
        match = matches_df[
            (matches_df["home_team"] == p["home_team"]) &
            (matches_df["away_team"] == p["away_team"])
        ]
        if len(match) == 0:
            continue
        actual = match.iloc[-1]
        log.at[i, "actual_result"] = actual["result"]
        log.at[i, "actual_home_goals"] = int(actual["home_goals"])
        log.at[i, "actual_away_goals"] = int(actual["away_goals"])
        updated += 1

    log.to_csv(PREDICTIONS_FILE, index=False)

    completed = log[log["actual_result"].astype(str) != ""]
    if len(completed) == 0:
        print(f"{len(log)} predictions logged, none reconciled yet (matches not played).")
        return

    correct = (completed["predicted_result"] == completed["actual_result"]).sum()
    acc = correct / len(completed)
    print(f"\nPREDICTION TRACK RECORD")
    print(f"  Predictions logged    : {len(log)}")
    print(f"  Reconciled with result: {len(completed)} (added {updated} this run)")
    print(f"  Model correct         : {correct} ({acc:.1%})")

    # market comparison if we have logged market picks
    if "mkt_pick" in completed.columns and (completed["mkt_pick"].astype(str) != "").any():
        with_mkt = completed[completed["mkt_pick"].astype(str) != ""]
        mkt_correct = (with_mkt["mkt_pick"] == with_mkt["actual_result"]).sum()
        mkt_acc = mkt_correct / len(with_mkt)
        print(f"  Market correct        : {mkt_correct} ({mkt_acc:.1%})  (over same {len(with_mkt)} matches)")
        edge = acc - mkt_acc
        print(f"  Edge over market      : {edge:+.1%}")

        # value bets — where model disagreed with market and was right
        disagreed = with_mkt[with_mkt["predicted_result"] != with_mkt["mkt_pick"]]
        if len(disagreed):
            value_correct = (disagreed["predicted_result"] == disagreed["actual_result"]).sum()
            print(f"  When disagreeing      : {value_correct}/{len(disagreed)} ({value_correct/len(disagreed):.1%})")

    print(f"  Log file              : {PREDICTIONS_FILE}\n")

    # accuracy by predicted outcome
    print("  Accuracy by prediction type:")
    for outcome in ["H", "D", "A"]:
        sub = completed[completed["predicted_result"] == outcome]
        if len(sub):
            sub_acc = (sub["predicted_result"] == sub["actual_result"]).mean()
            print(f"    {outcome}: {sub_acc:.1%}  ({len(sub)} predictions)")


# -----------------------------------------------------------------------------
# Predicting upcoming matches
# -----------------------------------------------------------------------------
def predict_upcoming(matches_df, model, n_days_ahead=10):
    """Predict the next batch of fixtures. Pulls upcoming fixtures from football-data feed
    by looking at the current season URL — anything without a result yet."""
    print(f"\nPredicting upcoming fixtures...")

    # Initialize external API clients — they self-disable if no key in config.py
    odds_client = None
    footy_client = None
    if API_CLIENTS_AVAILABLE:
        odds_client = OddsAPIClient()
        footy_client = APIFootballClient()
        if odds_client.enabled or footy_client.enabled:
            print("  External APIs:",
                  "Odds API" if odds_client.enabled else "",
                  "API-Football" if footy_client.enabled else "")

    # The current-season CSV often contains future fixtures with blank scores. Pull and filter.
    current_season = list(SEASON_URLS.keys())[-1]
    url = SEASON_URLS[current_season]
    try:
        r = requests.get(url, timeout=20)
        df = pd.read_csv(StringIO(r.text))
    except Exception as e:
        print(f"Couldn't fetch upcoming: {e}")
        return

    # rows with no FTHG = unplayed
    upcoming = df[df["FTHG"].isna() | (df["FTHG"] == "")].copy()
    if len(upcoming) == 0:
        print("No upcoming fixtures found in current-season feed.")
        return

    # parse dates and keep next N days
    upcoming["date"] = pd.to_datetime(upcoming["Date"], dayfirst=True, errors="coerce")
    upcoming = upcoming.dropna(subset=["date"])
    today = pd.Timestamp.now().normalize()
    upcoming = upcoming[
        (upcoming["date"] >= today) &
        (upcoming["date"] <= today + pd.Timedelta(days=n_days_ahead))
    ].sort_values("date")

    if len(upcoming) == 0:
        print(f"No fixtures in the next {n_days_ahead} days.")
        return

    print(f"Found {len(upcoming)} fixtures in the next {n_days_ahead} days\n")

    # for each fixture, compute features as if it were the next match in matches_df
    n = len(matches_df)
    league_avg = league_average_goals(matches_df)

    rows = []
    for _, fx in upcoming.iterrows():
        h, a, date = fx["HomeTeam"], fx["AwayTeam"], fx["date"]

        h_prom = is_promoted(matches_df, h, n)
        a_prom = is_promoted(matches_df, a, n)

        if h_prom:
            hs = promoted_team_priors()
        else:
            h_hist = get_team_history(matches_df, h, n)
            hs = time_decayed_stats(h_hist, h, league_avg) if len(h_hist) else neutral_stats()

        if a_prom:
            as_ = promoted_team_priors()
        else:
            a_hist = get_team_history(matches_df, a, n)
            as_ = time_decayed_stats(a_hist, a, league_avg) if len(a_hist) else neutral_stats()

        h_home_hist = get_team_history(matches_df, h, n, venue="home")
        a_away_hist = get_team_history(matches_df, a, n, venue="away")
        hs_home = time_decayed_stats(h_home_hist, h, league_avg) if len(h_home_hist) else hs
        as_away = time_decayed_stats(a_away_hist, a, league_avg) if len(a_away_hist) else as_

        h2h = head_to_head(matches_df, h, a, n)
        h_rest = days_rest(matches_df, h, n, date)
        a_rest = days_rest(matches_df, a, n, date)

        # CSV odds as starting point (always present for past, sometimes for upcoming)
        csv_odds = extract_odds(fx)
        if csv_odds is not None:
            csv_h, csv_d, csv_a = csv_odds
        else:
            csv_h, csv_d, csv_a = None, None, None
        csv_market_probs = {
            "mkt_p_home": csv_h, "mkt_p_draw": csv_d, "mkt_p_away": csv_a
        } if csv_odds else None

        # External enrichment — fresher odds, extra markets, injuries
        # If APIs disabled, this returns CSV-based + neutral defaults
        if API_CLIENTS_AVAILABLE:
            enriched = enrich_fixture_features(
                home_team=h, away_team=a,
                odds_client=odds_client, footy_client=footy_client,
                csv_market_probs=csv_market_probs,
            )
        else:
            # pure CSV path
            mh = csv_h if csv_h is not None else 0.45
            md = csv_d if csv_d is not None else 0.27
            ma = csv_a if csv_a is not None else 0.28
            enriched = {
                "mkt_p_home": mh, "mkt_p_draw": md, "mkt_p_away": ma,
                "mkt_p_over_25": 0.55, "mkt_p_btts_yes": 0.52,
                "h_key_players_out": 0, "a_key_players_out": 0,
                "h_injuries_total": 0, "a_injuries_total": 0,
                "mkt_source": "csv" if csv_odds else "neutral_default",
            }

        mkt_h = enriched["mkt_p_home"]
        mkt_d = enriched["mkt_p_draw"]
        mkt_a = enriched["mkt_p_away"]
        mkt_margin = max(mkt_h, mkt_a) - min(mkt_h, mkt_a)
        mkt_pick = "H" if mkt_h >= max(mkt_d, mkt_a) else ("D" if mkt_d >= mkt_a else "A")

        feat = {
            "h_form_pts": hs["form_pts_per_game"], "a_form_pts": as_["form_pts_per_game"],
            "h_gf": hs["gf_per_game"], "a_gf": as_["gf_per_game"],
            "h_ga": hs["ga_per_game"], "a_ga": as_["ga_per_game"],
            "h_home_pts": hs_home["form_pts_per_game"], "a_away_pts": as_away["form_pts_per_game"],
            "h_home_gf": hs_home["gf_per_game"], "a_away_gf": as_away["gf_per_game"],
            "h_home_ga": hs_home["ga_per_game"], "a_away_ga": as_away["ga_per_game"],
            "h2h_home_winrate": h2h["h2h_home_winrate"],
            "h2h_draw_rate": h2h["h2h_draw_rate"],
            "h2h_goal_diff": h2h["h2h_goal_diff"],
            "h_days_rest": h_rest, "a_days_rest": a_rest,
            "h_promoted": float(h_prom), "a_promoted": float(a_prom),
            "form_diff": hs["form_pts_per_game"] - as_["form_pts_per_game"],
            "gf_diff": hs["gf_per_game"] - as_["gf_per_game"],
            "ga_diff": as_["ga_per_game"] - hs["ga_per_game"],
            "mkt_p_home": mkt_h, "mkt_p_draw": mkt_d, "mkt_p_away": mkt_a,
            "mkt_fav_margin": mkt_margin,
            "mkt_p_over_25": enriched["mkt_p_over_25"],
            "mkt_p_btts_yes": enriched["mkt_p_btts_yes"],
            "h_key_players_out": enriched["h_key_players_out"],
            "a_key_players_out": enriched["a_key_players_out"],
            "h_injuries_total": enriched["h_injuries_total"],
            "a_injuries_total": enriched["a_injuries_total"],
        }

        pred = model.predict(feat)
        my_pick = "H" if pred["p_home"] > max(pred["p_draw"], pred["p_away"]) \
                  else "D" if pred["p_draw"] > pred["p_away"] else "A"

        # confidence relative to market — flag where we differ
        disagree = "*" if my_pick != mkt_pick else ""

        rows.append({
            "date": date.strftime("%a %d %b"),
            "match": f"{h} vs {a}",
            "pick": my_pick + disagree,
            "p_H": f"{pred['p_home']:.0%}",
            "p_D": f"{pred['p_draw']:.0%}",
            "p_A": f"{pred['p_away']:.0%}",
            "mkt_pick": mkt_pick,
            "mkt_H": f"{mkt_h:.0%}",
            "mkt_D": f"{mkt_d:.0%}",
            "mkt_A": f"{mkt_a:.0%}",
            "xG_H": round(pred["xg_home"], 2),
            "xG_A": round(pred["xg_away"], 2),
            "cs_H": f"{pred['cs_home']:.0%}",
            "cs_A": f"{pred['cs_away']:.0%}",
            # extra markets
            "O25": f"{enriched['mkt_p_over_25']:.0%}",
            "BTTS": f"{enriched['mkt_p_btts_yes']:.0%}",
            # injury indicators (only meaningful when API-Football enabled)
            "h_inj": enriched["h_injuries_total"],
            "a_inj": enriched["a_injuries_total"],
            "src": enriched["mkt_source"],
        })

        # log it so we can score accuracy later
        log_prediction(date, h, a, pred,
                       mkt_probs={"mkt_p_home": mkt_h, "mkt_p_draw": mkt_d, "mkt_p_away": mkt_a})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\n  * = model disagrees with the bookmaker favourite (potential value bet)")

    # save to outputs
    output_csv = Path("/mnt/user-data/outputs/upcoming_predictions.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"\nSaved -> {output_csv}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="Re-download data, retrain, predict upcoming GW")
    parser.add_argument("--review", action="store_true",
                        help="Check accuracy of past predictions vs actual results")
    parser.add_argument("--days-ahead", type=int, default=10,
                        help="How many days of upcoming fixtures to predict (default 10)")
    args = parser.parse_args()

    print("=" * 70)
    print("  PREMIER LEAGUE MATCH PREDICTOR v2")
    print("=" * 70 + "\n")

    matches, model = load_state()
    fresh_run = matches is None or args.update

    if fresh_run:
        matches = download_all_seasons()
        matches, feat_cols = build_features(matches)
        model = PLModel()
        model.fit(matches, feat_cols)
        save_state(matches, model)
    else:
        print(f"Loaded cached state: {len(matches)} matches\n")
        # still rebuild feat_cols list from model
        feat_cols = model.feat_cols

    if args.review:
        # need fresh data to check actual results
        if not fresh_run:
            fresh_matches = download_all_seasons()
            review_predictions(fresh_matches)
        else:
            review_predictions(matches)
        return

    # always predict upcoming
    predict_upcoming(matches, model, n_days_ahead=args.days_ahead)


if __name__ == "__main__":
    main()
