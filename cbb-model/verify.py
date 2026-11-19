"""Audit the backtest: every prediction next to the real result, plus checks
for the usual ways a backtest fools you.

    python verify.py        # run after train.py; writes backtest_predictions.csv

Checks:
  1. No look-ahead: recompute ratings with all games from a cutoff date onward
     deleted, and confirm the features for games on that date are identical.
  2. Weak opponents: accuracy on Division I vs Division I games only, since
     blowouts of small non-D-I schools are easy to predict.
  3. Spot checks: random games and famous upsets, with team names and scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cbb.data import DATA_DIR, load_games
from cbb.ratings import FEATURES, build_features
from train import fit_models, predict_proba

feats = pd.read_parquet(DATA_DIR / "features.parquet")
done = feats[feats["completed"] & (feats["season"] > feats["season"].min())]

# --- Re-run the walk-forward backtest, keeping every prediction ---
preds = []
for season in sorted(done["season"].unique())[3:]:
    models = fit_models(done[done["season"] < season])
    test = done[done["season"] == season].copy()
    test["p_home"] = predict_proba(models, test)["ensemble"]
    preds.append(test)
preds = pd.concat(preds, ignore_index=True)
preds["predicted_winner"] = np.where(preds["p_home"] >= 0.5, preds["home_team"], preds["away_team"])
preds["actual_winner"] = np.where(preds["home_win"] == 1, preds["home_team"], preds["away_team"])
preds["correct"] = preds["predicted_winner"] == preds["actual_winner"]
cols = ["date", "season", "away_team", "home_team", "neutral", "p_home",
        "predicted_winner", "away_score", "home_score", "actual_winner", "correct"]
out = DATA_DIR.parent / "backtest_predictions.csv"
preds[cols].to_csv(out, index=False)
print(f"Wrote {len(preds):,} predictions to {out.name}")
print(f"Overall: {preds['correct'].mean():.1%} correct\n")

# --- 1. Look-ahead check ---
print("1. Look-ahead check")
games = load_games(list(range(2022, 2027)), refresh_latest=False)
cutoff = pd.Timestamp("2025-02-15")
full, _ = build_features(games)
cut, _ = build_features(games[games["date"] < cutoff + pd.Timedelta(days=1)])
day = lambda f: f[f["date"].dt.normalize() == cutoff].set_index("game_id")[FEATURES].sort_index()
a, b = day(full), day(cut)
same = np.allclose(a.to_numpy(), b.loc[a.index].to_numpy())
print(f"   {len(a)} games on {cutoff.date()}: features with later games deleted are "
      f"{'IDENTICAL' if same else 'DIFFERENT (leak!)'}\n")

# --- 2. Division I vs Division I only ---
print("2. Division I vs Division I games only")
# A team is treated as D-I in a season if it played 20+ games that season;
# non-D-I schools appear only a handful of times.
counts = pd.concat([feats[feats["completed"]][["season", "home_id"]].rename(columns={"home_id": "t"}),
                    feats[feats["completed"]][["season", "away_id"]].rename(columns={"away_id": "t"})])
counts = counts.value_counts()
d1 = set(counts[counts >= 20].index)
is_d1 = [((s, h) in d1) and ((s, a) in d1) for s, h, a in
         zip(preds["season"], preds["home_id"], preds["away_id"])]
dd = preds[is_d1]
print(f"   all games:        {preds['correct'].mean():.1%} of {len(preds):,}")
print(f"   D-I vs D-I only:  {dd['correct'].mean():.1%} of {len(dd):,}")
print(f"   2027 (so far), D-I vs D-I: {dd[dd.season == 2027]['correct'].mean():.1%} "
      f"of {(dd.season == 2027).sum()}")
fav = np.maximum(dd["p_home"], 1 - dd["p_home"])
for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
    m = (fav >= lo) & (fav < hi)
    print(f"   favorite given {lo:.0%}-{hi:.0%}: won {dd[m]['correct'].mean():.1%} "
          f"({m.sum():,} games)")

# --- 3. Spot checks ---
print("\n3. Ten random D-I games")
show = lambda d: print(d.assign(p_home=d["p_home"].map("{:.0%}".format))[
    ["date", "away_team", "away_score", "home_team", "home_score", "p_home", "correct"]
].to_string(index=False))
show(dd.sample(10, random_state=1).sort_values("date"))
print("\nFamous NCAA tournament upsets (model should have picked the favorite)")
upsets = [("UMBC", "Virginia Cavaliers", "2018-03-16"),
          ("Fairleigh Dickinson", "Purdue", "2023-03-17"),
          ("Saint Peter's", "Kentucky", "2022-03-17")]
for winner, loser, d in upsets:
    g = preds[(preds["date"].dt.strftime("%Y-%m-%d") == d)
              & preds["home_team"].str.contains("|".join([winner, loser]))
              & preds["away_team"].str.contains("|".join([winner, loser]))]
    if len(g):
        show(g)
