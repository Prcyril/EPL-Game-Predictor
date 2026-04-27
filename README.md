# PL Predictor

Predicts Premier League match outcomes with continued learning. Built around a simple loop: predict before each gameweek, log predictions, retrain when results come in, track accuracy over time.

## Files

- `pl_predictor.py` — main predictor, run this
- `api_clients.py` — wrappers for The Odds API and API-Football
- `config_template.py` — copy to `config.py` and add your API keys
- `.gitignore` — keeps `config.py` out of git

## Quick start (no API keys needed)

```bash
pip install pandas numpy scikit-learn requests pyarrow
python3 pl_predictor.py
```

This trains from football-data.co.uk CSVs and uses the betting odds embedded in those CSVs. You'll get match predictions vs market benchmark right away.

## Add external APIs (optional but recommended)

```bash
cp config_template.py config.py
# edit config.py — paste your API keys
python3 api_clients.py    # smoke test
python3 pl_predictor.py --update
```

Sign-ups (both free tiers, no credit card):
- https://the-odds-api.com — 500 requests/month, gives you live pre-match odds for 1X2 / O/U 2.5 / BTTS
- https://www.api-football.com — 100 requests/day, gives you current injuries and predicted lineups

## Weekly workflow

```bash
# Friday morning — get next gameweek's predictions
python3 pl_predictor.py --update

# Sunday/Monday — review last week's accuracy vs market
python3 pl_predictor.py --review
```

## Reading the output

```
match               pick  p_H  p_D  p_A  mkt_pick  mkt_H  mkt_D  mkt_A   O25   BTTS  src
Liverpool vs Leeds  H     72%  18%  10%  H         78%    14%    8%     65%   55%   odds_api
Brighton vs Wolves  D*    32%  38%  30%  H         44%    28%    28%    58%   52%   odds_api
```

- `*` next to your pick means model disagrees with market — potential value bet
- `src=odds_api` means live odds, `src=csv` means CSV fallback, `src=neutral_default` means no odds available
- `O25` = probability of Over 2.5 goals (market's view)
- `BTTS` = probability both teams to score (market's view)

## What gets cached

- `pl_data/matches.parquet` — all historical match data
- `pl_data/model.pkl` — trained model
- `pl_data/predictions_log.csv` — every prediction with market comparison
- `pl_data/api_cache/` — recent API responses (saves quota)
- `pl_data/meta.json` — training history with CV scores
