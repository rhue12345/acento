# College basketball win predictor

Predicts the winner (win probability and point spread) of men's Division I college basketball games.

## Quick start

```bash
pip install -r requirements.txt
python train.py              # downloads 2008–today, backtests, saves model.pkl (~1 min)
python predict.py            # next 3 days of games
python predict.py --days 7
python predict.py --matchup "Duke" "Kansas" --neutral   # first team is home
python predict.py --top 25   # current power ratings
```

`predict.py` re-downloads the current season each run, so ratings include last night's results. Rerun `train.py` occasionally (e.g. weekly) to refit the model on new games.

## How it works

1. **Data** (`cbb/data.py`): every game since 2008 from the public [sportsdataverse hoopR-mbb-data](https://github.com/sportsdataverse/hoopR-mbb-data) mirror of ESPN results, updated daily.
2. **Ratings** (`cbb/ratings.py`), computed for each game using only games from earlier dates, so nothing leaks from the future:
   - **Elo** with a margin-of-victory multiplier and home court, carried over between seasons (65% kept).
   - **Adjusted margin rating** (in the spirit of KenPom/Massey): a ridge regression of each game's margin (capped at 25) on team strengths plus home court, refit every game day, weighting recent games more and shrinking each team toward a preseason prior from last season.
3. **Models** (`train.py`): a logistic regression and a gradient-boosted classifier on the rating differences, home/neutral, and an "early season" adjustment that learns how much to trust each rating while teams have played few games. The final win probability averages the two. A linear model gives the projected margin.

## Backtest

Walk-forward: each season is predicted by a model trained only on earlier seasons. Seasons 2012–2027, 89,485 real games. `python verify.py` reproduces this and writes every prediction next to the actual result in `backtest_predictions.csv`.

| Method | All games | D-I vs D-I only (80,918 games) |
|---|---|---|
| Always pick the home team | 67.8% (home games only) | – |
| Pick the higher-Elo team | 70.1% | – |
| **This model** | **74.4%** | **72.5%** |

The **72.5%** is the honest number for games between Division I teams. The 74.4% is inflated by easy wins over small non-D-I schools.

Checks in `verify.py`:
- **No look-ahead:** ratings recomputed with every game after a cutoff date deleted give identical features for that date's games.
- **Calibration (D-I vs D-I):** favorites given 60–70% won 64.0%; 70–80% won 72.8%; 80–90% won 83.7%; 90%+ won 93.9%.
- **Postseason** (conference + NCAA tournaments): 67.5% — closer-matched teams.
- Spread error (mean absolute) is about 10 points.
- **Caveat:** the rating settings (Elo K, carryover, prior strength, etc.) were picked by hand, not tuned on the test seasons, but they haven't been checked against a separate holdout either.

The model hasn't been compared with betting lines yet. That comparison would show whether it is competitive.

## Ideas to improve it

- Use box scores (possessions, offensive/defensive efficiency, four factors) instead of raw margins. Team box scores are in the same data source (`mbb/team_box`).
- Preseason priors from returning minutes, transfers and recruiting instead of last season's rating alone.
- Rest days, travel distance, and injuries.
- Compare against betting lines to look for edges.
