"""Can a stronger model find the real favorite in games the baseline calls a toss-up?

    python tossups.py            # tune on 2012-2019, report on 2020-2026
    python tossups.py --no-tune  # skip the tuning grid

Baseline = the model in train.py. Its toss-ups are D-I vs D-I games where its
projected spread is under 1 point (also reported: under 2). That game set is
fixed before the new model is scored, so the new model can't look better by
moving games in or out of the toss-up group.

New model = baseline features + possession-based ratings + rest. Its settings
are tuned only on 2012-2019; 2020-2026 is never used for any choice and is the
honest test.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

from cbb.data import DATA_DIR, load_games
from cbb.efficiency import EffParams, build_eff_features, load_box, rest_features
from cbb.ratings import FEATURES as BASE

TUNE = range(2012, 2020)
TEST = range(2020, 2027)
NEW = BASE + ["eff_margin", "eff_margin_early", "rest_diff", "b2b_diff"]


def fit(train: pd.DataFrame, feats: list[str]):
    lr = LogisticRegression(fit_intercept=False, max_iter=2000).fit(train[feats], train["home_win"])
    flip = train[feats].mul([1 if f == "home" else -1 for f in feats], axis=1)
    flip["home"] = -train["home"]
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                        min_samples_leaf=200, l2_regularization=1.0)
    gb.fit(pd.concat([train[feats], flip]),
           np.r_[train["home_win"], 1 - train["home_win"]])
    spread = LinearRegression(fit_intercept=False).fit(train[feats], train["margin"])
    return lr, gb, spread


def walk_forward(df: pd.DataFrame, feats: list[str], seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        lr, gb, spread = fit(df[df["season"] < s], feats)
        t = df[df["season"] == s].copy()
        t["p"] = (lr.predict_proba(t[feats])[:, 1] + gb.predict_proba(t[feats])[:, 1]) / 2
        t["spread"] = spread.predict(t[feats])
        out.append(t)
    return pd.concat(out)


def add_new_features(df: pd.DataFrame, eff: pd.DataFrame, rest: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in eff.columns if c != "game_id" and c in df], errors="ignore")
    df = df.merge(eff, on="game_id", how="left").merge(rest, on="game_id", how="left")
    df["eff_margin"] = df["eff_margin"].fillna(df["rating_diff"])
    early = np.exp(-np.minimum(df["home_gp"], df["away_gp"]) / 5.0)
    df["eff_margin_early"] = df["eff_margin"] * early
    df["rest_diff"] = df["rest_diff"].fillna(0)
    df["b2b_diff"] = (df["home_b2b"] - df["away_b2b"]).fillna(0)
    return df


def d1_mask(df: pd.DataFrame) -> np.ndarray:
    c = pd.concat([df[["season", "home_id"]].set_axis(["s", "t"], axis=1),
                   df[["season", "away_id"]].set_axis(["s", "t"], axis=1)]).value_counts()
    d1 = set(c[c >= 20].index)
    return np.array([((s, h) in d1) and ((s, a) in d1)
                     for s, h, a in zip(df["season"], df["home_id"], df["away_id"])])


def ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()

    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    games = load_games(list(range(2008, 2028)), refresh_latest=False)
    box = load_box(list(range(2008, 2028)), refresh_latest=False)
    rest = rest_features(games[games["completed"] | ~games["completed"]])
    done = feats[feats["completed"] & (feats["season"] > 2008)].copy()
    done = done[d1_mask(done)]  # D-I vs D-I only, everywhere below

    # --- Tune possession-rating settings on 2012-2019 only ---
    best = EffParams()
    if not args.no_tune:
        print("Tuning on 2012-2019 (log loss, lower is better):")
        scores = {}
        for hl, pg in itertools.product([30.0, 60.0, 120.0], [2.0, 4.0, 8.0]):
            p = EffParams(half_life=hl, prior_games=pg)
            df = add_new_features(done, build_eff_features(games, box, p), rest)
            wf = walk_forward(df[df["season"] < 2020], NEW, TUNE)
            scores[p] = ll(wf["home_win"], wf["p"])
            print(f"  half-life {hl:5.0f} days, prior {pg:3.0f} games: {scores[p]:.4f}")
        best = min(scores, key=scores.get)
        print(f"  chosen: half-life {best.half_life:.0f}, prior {best.prior_games:.0f}\n")

    df = add_new_features(done, build_eff_features(games, box, best), rest)
    base = walk_forward(df, BASE, TEST)
    new = walk_forward(df, NEW, TEST).set_index("game_id").loc[base["game_id"]].reset_index()
    base = base.reset_index(drop=True)
    y = base["home_win"].to_numpy()
    base_ok = (base["p"] >= 0.5) == (y == 1)
    new_ok = (new["p"] >= 0.5).to_numpy() == (y == 1)

    print(f"Test seasons 2020-2026, D-I vs D-I: {len(base):,} games")
    print(f"  all games   baseline {base_ok.mean():.1%}  log loss {ll(y, base['p']):.4f}")
    print(f"              new      {new_ok.mean():.1%}  log loss {ll(y, new['p']):.4f}")

    for width in (1.0, 2.0):
        m = (base["spread"].abs() < width).to_numpy()
        n = m.sum()
        ci = 1.96 * np.sqrt(0.25 / n)
        print(f"\nBaseline toss-ups (baseline spread under {width:.0f} pt): {n:,} games")
        print(f"  baseline picks right: {base_ok[m].mean():.1%}")
        print(f"  new model picks right: {new_ok[m].mean():.1%}   "
              f"(coin flip would land {0.5 - ci:.1%}-{0.5 + ci:.1%})")
        conf = np.maximum(new["p"], 1 - new["p"]).to_numpy()
        for cut in (0.55, 0.60, 0.65):
            k = m & (conf >= cut)
            if k.sum() >= 30:
                print(f"  new model says {cut:.0%}+ (not a toss-up): {k.sum():5,} games, "
                      f"new right {new_ok[k].mean():.1%} (expected {conf[k].mean():.1%}), "
                      f"baseline right {base_ok[k].mean():.1%}")
        flips = m & ((new["p"] >= 0.5).to_numpy() != (base["p"] >= 0.5).to_numpy())
        print(f"  games where the new model picks the other team: {flips.sum():,}, "
              f"new right {new_ok[flips].mean():.1%}")

    out = base[["date", "away_team", "home_team", "home_score", "away_score", "spread", "p"]].rename(
        columns={"spread": "base_spread", "p": "base_p_home"})
    out["new_spread"] = new["spread"].to_numpy()
    out["new_p_home"] = new["p"].to_numpy()
    out["base_correct"], out["new_correct"] = base_ok, new_ok
    out.to_csv("tossup_results.csv", index=False)
    print("\nPer-game results: tossup_results.csv")


if __name__ == "__main__":
    main()
