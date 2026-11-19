"""Does knowing who's playing help with toss-ups?

    python availability.py

Same fixed toss-up set and 2020-2026 test seasons as tossups.py. Compares:
  baseline        the model in train.py
  efficiency      + possession ratings and rest (tossups.py, tuned settings)
  + last game     + key players who sat out each team's previous game (usable pre-game)
  + this game     + key players who did not play in this game (UPPER BOUND ONLY:
                    taken from the game's own box score, i.e. a perfect injury report)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cbb.data import DATA_DIR, load_games
from cbb.efficiency import EffParams, build_eff_features, load_box, rest_features
from cbb.players import availability_features, game_level, load_player_box
from cbb.ratings import FEATURES as BASE
from tossups import NEW, TEST, add_new_features, d1_mask, ll, walk_forward

SEASONS = list(range(2008, 2028))
SETS = {
    "baseline": BASE,
    "efficiency": NEW,
    "+ last game": NEW + ["miss_last_diff"],
    "+ this game (upper bound)": NEW + ["miss_last_diff", "miss_now_diff"],
}


def main() -> None:
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    games = load_games(SEASONS, refresh_latest=False)
    box = load_box(SEASONS, refresh_latest=False)
    print("Loading player box scores ...")
    pb = load_player_box(SEASONS, refresh_latest=False)
    print("Building availability features ...")
    av = game_level(availability_features(pb), games)

    done = feats[feats["completed"] & (feats["season"] > 2008)].copy()
    done = done[d1_mask(done)]
    eff = build_eff_features(games, box, EffParams(half_life=120.0, prior_games=2.0))
    df = add_new_features(done, eff, rest_features(games)).merge(av, on="game_id", how="left")
    df[["miss_last_diff", "miss_now_diff"]] = df[["miss_last_diff", "miss_now_diff"]].fillna(0)

    t = df[df["season"].isin(TEST)]
    print(f"\nHow often a key player is missing (either team) (test seasons, {len(t):,} games):")
    print(f"  a key player sat out the previous game: {(t['miss_last_diff'] != 0).mean():.1%}")
    print(f"  a key player sits out this game:        {(t['miss_now_diff'] != 0).mean():.1%}")

    res = {name: walk_forward(df, f, TEST).set_index("game_id") for name, f in SETS.items()}
    ids = res["baseline"].index
    res = {k: v.loc[ids] for k, v in res.items()}
    y = res["baseline"]["home_win"].to_numpy()
    ok = {k: ((v["p"] >= 0.5).to_numpy() == (y == 1)) for k, v in res.items()}
    base_spread = res["baseline"]["spread"].abs().to_numpy()

    print(f"\nAll {len(y):,} D-I vs D-I games, 2020-2026:")
    for k, v in res.items():
        print(f"  {k:27s} {ok[k].mean():.1%}  log loss {ll(y, v['p']):.4f}")

    for width in (1.0, 2.0):
        m = base_spread < width
        n = m.sum()
        ci = 1.96 * np.sqrt(0.25 / n)
        print(f"\nBaseline toss-ups (spread under {width:.0f} pt): {n:,} games "
              f"(coin flip lands {0.5 - ci:.1%}-{0.5 + ci:.1%})")
        for k, v in res.items():
            flips = m & ((v["p"] >= 0.5).to_numpy() != (res["baseline"]["p"] >= 0.5).to_numpy())
            extra = (f"  picks the other team in {flips.sum():,}, right {ok[k][flips].mean():.1%}"
                     if flips.sum() else "")
            print(f"  {k:27s} {ok[k][m].mean():.1%}  log loss {ll(y[m], v['p'].to_numpy()[m]):.4f}{extra}")

    # Toss-ups where one side is missing real production.
    m = (base_spread < 2) & (res["baseline"]["miss_now_diff"].abs().to_numpy() >= 5)
    print(f"\nToss-ups (under 2 pt) where one team is missing 5+ Game Score of key players "
          f"in this game: {m.sum():,} games")
    for k in res:
        print(f"  {k:27s} {ok[k][m].mean():.1%}")
    m = res["baseline"]["miss_now_diff"].abs().to_numpy() >= 8
    print(f"\nAll games where one team is missing 8+ Game Score of key players: {m.sum():,}")
    for k in res:
        print(f"  {k:27s} {ok[k][m].mean():.1%}")
    m = (base_spread < 2) & (res["baseline"]["miss_last_diff"].abs().to_numpy() >= 5)
    print(f"Toss-ups (under 2 pt) where one team was missing 5+ in its previous game: {m.sum():,} games")
    for k in res:
        print(f"  {k:27s} {ok[k][m].mean():.1%}")


if __name__ == "__main__":
    main()
