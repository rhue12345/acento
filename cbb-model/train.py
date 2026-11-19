"""Build features, backtest season by season, and save the final model.

    python train.py                      # seasons 2008 through the current one
    python train.py --first-season 2015  # faster, less history
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from cbb.data import DATA_DIR, load_games
from cbb.ratings import FEATURES, build_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
REPORT_PATH = ROOT / "backtest.json"


def current_season(today: date | None = None) -> int:
    """Season label = the year it ends (Nov 2026 -> 2027)."""
    today = today or date.today()
    return today.year + 1 if today.month >= 8 else today.year


def symmetric(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Add every game a second time from the away team's view.

    This keeps the gradient-boosted model symmetric: swapping home and away
    on a neutral court flips the probability exactly.
    """
    flipped = df[FEATURES].copy()
    for c in FEATURES:
        if c != "home":
            flipped[c] = -flipped[c]
    flipped["home"] = -df["home"]
    X = pd.concat([df[FEATURES], flipped], ignore_index=True)
    y = np.concatenate([df["home_win"].to_numpy(), 1 - df["home_win"].to_numpy()])
    return X, y


def make_models() -> dict:
    return {
        # No intercept: with home=0 the model gives 50% to equal teams.
        "logistic": LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000),
        "boosted": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=200, l2_regularization=1.0,
        ),
    }


def fit_models(train: pd.DataFrame) -> dict:
    models = make_models()
    models["logistic"].fit(train[FEATURES], train["home_win"])
    Xs, ys = symmetric(train)
    models["boosted"].fit(Xs, ys)
    spread = LinearRegression(fit_intercept=False).fit(train[FEATURES], train["margin"])
    models["spread"] = spread
    return models


def predict_proba(models: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    p_lr = models["logistic"].predict_proba(df[FEATURES])[:, 1]
    p_gb = models["boosted"].predict_proba(df[FEATURES])[:, 1]
    return {"logistic": p_lr, "boosted": p_gb, "ensemble": (p_lr + p_gb) / 2}


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y, p > 0.5),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-season", type=int, default=2008)
    ap.add_argument("--last-season", type=int, default=current_season())
    ap.add_argument("--min-train-seasons", type=int, default=3)
    args = ap.parse_args()

    seasons = list(range(args.first_season, args.last_season + 1))
    print(f"Loading seasons {seasons[0]}-{seasons[-1]} ...")
    games = load_games(seasons)
    print(f"  {games['completed'].sum():,} completed games, "
          f"{(~games['completed']).sum():,} scheduled")

    print("Building point-in-time ratings (takes a few minutes) ...")
    feats, _ = build_features(games)
    feats.to_parquet(DATA_DIR / "features.parquet")

    done = feats[feats["completed"]].copy()
    # The first season has no prior-season information; use it only as warm-up.
    done = done[done["season"] > seasons[0]]

    # --- Walk-forward backtest: train on past seasons, test on the next one ---
    rows, all_preds = [], []
    test_seasons = sorted(done["season"].unique())[args.min_train_seasons:]
    for season in test_seasons:
        train = done[done["season"] < season]
        test = done[done["season"] == season]
        models = fit_models(train)
        probs = predict_proba(models, test)
        y = test["home_win"].to_numpy()
        row = {"season": int(season), "games": int(len(test))}
        # Baselines: pick the home team; pick the higher-Elo team.
        row["home_team_acc"] = float(np.mean(y[test["home"] == 1] == 1))
        row["elo_only_acc"] = float(np.mean((test["elo_diff"] > 0) == (y == 1)))
        for name, p in probs.items():
            for k, v in metrics(y, p).items():
                row[f"{name}_{k}"] = float(v)
        spread = models["spread"].predict(test[FEATURES])
        row["spread_mae"] = float(np.mean(np.abs(spread - test["margin"])))
        rows.append(row)
        all_preds.append(test.assign(p_home=probs["ensemble"]))
        print(f"  {season}: {len(test):5d} games  "
              f"acc {row['ensemble_accuracy']:.3f}  "
              f"log loss {row['ensemble_log_loss']:.3f}  "
              f"spread MAE {row['spread_mae']:.1f}")

    report = pd.DataFrame(rows)
    preds = pd.concat(all_preds, ignore_index=True)
    y, p = preds["home_win"].to_numpy(), preds["p_home"].to_numpy()
    overall = metrics(y, p)
    tourney = preds[preds["season_type"] == 3]
    tourney_m = metrics(tourney["home_win"].to_numpy(), tourney["p_home"].to_numpy())

    # Calibration: when we say 70%, does the favorite win ~70% of the time?
    fav_p = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    bins = pd.cut(fav_p, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    calib = (pd.DataFrame({"bin": bins, "pred": fav_p, "won": fav_won})
             .groupby("bin", observed=True).agg(games=("won", "size"),
                                                predicted=("pred", "mean"),
                                                actual=("won", "mean")))

    print("\nAll backtest seasons (ensemble):")
    print(f"  accuracy {overall['accuracy']:.3f}  log loss {overall['log_loss']:.3f}  "
          f"Brier {overall['brier']:.3f}")
    print(f"  postseason games ({len(tourney)}): accuracy {tourney_m['accuracy']:.3f}")
    print("\nCalibration (favorite's predicted vs actual win rate):")
    print(calib.round(3).to_string())

    REPORT_PATH.write_text(json.dumps({
        "overall": overall,
        "postseason": {**tourney_m, "games": int(len(tourney))},
        "by_season": rows,
        "calibration": calib.reset_index().astype({"bin": str}).to_dict("records"),
    }, indent=2))

    # --- Final model on every completed game ---
    final = fit_models(done)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"models": final, "features": FEATURES,
                     "trained_through": str(done["date"].max().date())}, f)
    print(f"\nSaved {MODEL_PATH.name} and {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
