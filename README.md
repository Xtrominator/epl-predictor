# EPL Match Outcome Predictor

My project using XGBoost to predict English Premier League results
(home win / draw / away win), then testing whether the predictions could
beat bookmaker odds.

## What it does
1. Downloads EPL data from [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (2017-18 to 2022-23 for training, 2023-24 for testing)
2. Builds features for each match:
   - Implied probabilities from Bet365 odds
   - Form (points in last 5 games)
   - Current league position of both teams
   - Unbeaten streaks
3. Tunes an XGBoost model with GridSearchCV
4. Compares accuracy against a simple "always pick the favorite" baseline
5. Backtests a value betting strategy on 2023-24 and plots the bankroll

## How to run
```
pip install -r requirements.txt
python epl_model.py
```
It needs internet since it downloads the CSVs. It saves `bankroll.png`.

## Results
Run the script and fill these in with your own numbers:

| | Accuracy |
|---|---|
| Favorite baseline | 59.7% |
| XGBoost model | 58.7% |

Betting backtest: bets = 277, wins = 47, final bankroll = 241.83 (started at 1000), ROI = -25.56%

## Things I learned / limitations
- Features only use matches *before* the one being predicted, otherwise the model sees the future.
- The odds are already a really strong feature, so beating the bookmakers is hard. A good result on one season could easily be luck.
- Bookmaker odds include a margin, so I normalised the implied probabilities.
- The backtest ignores things like betting limits and odds changing before kickoff.
- This is for learning, not real gambling advice.

## Findings
- The model (58.7%) did not beat the "pick the favorite" baseline (59.7%). The bookmaker odds already hold most of the useful information.
- The value betting strategy lost money on 2023-24: 277 bets, 47 wins, bankroll went from 1000 to about 242 (ROI -25.56%).
- This suggests the market was pretty efficient on this data. The model mostly bet on outsiders and draws where it disagreed with the odds, and the bookmakers were usually right.
- This is only one season, so the result is noisy either way.

