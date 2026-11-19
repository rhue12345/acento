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

Walk-forward: each season is predicted by a model trained only on earlier seasons. Seasons 2012–2027 (≈90,000 games):

| Method | Accuracy | Log loss |
|---|---|---|
| Always pick the home team | 67.8% (home games only) | – |
| Pick the higher-Elo team | 70.1% | – |
| Logistic regression | 74.3% | 0.509 |
| Gradient boosting | 74.5% | 0.504 |
| **Ensemble** | **74.4%** | **0.506** |

- Postseason (conference + NCAA tournaments): 67.5% accuracy — those games are between closer-matched teams.
- Calibration is good: when the favorite is given 70–80%, it wins 73.5% of the time; 90%+ favorites win 95%.
- Spread error (mean absolute) is about 10 points.

For context, Vegas closing lines pick roughly 75–76% of college basketball winners, so this is close to the practical ceiling for a results-only model. Full numbers are in `backtest.json`.

## Ideas to improve it

- Use box scores (possessions, offensive/defensive efficiency, four factors) instead of raw margins. Team box scores are in the same data source (`mbb/team_box`).
- Preseason priors from returning minutes, transfers and recruiting instead of last season's rating alone.
- Rest days, travel distance, and injuries.
- Compare against betting lines to look for edges.
