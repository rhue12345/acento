"""Run the pre-registered betting rules (see PREREGISTERED.md) on the holdout.

    python holdout_test.py      # after collect_lines.py and vs_vegas.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cbb.data import DATA_DIR

BREAK_EVEN = 110 / 210
FIRST_SAMPLE_LINES = 9772


def first_sample_ids() -> set[int]:
    ids = set()
    with (DATA_DIR / "lines_raw.jsonl").open() as f:
        for i, line in enumerate(f):
            if i >= FIRST_SAMPLE_LINES:
                break
            ids.add(json.loads(line)["game_id"])
    return ids


def judge(name: str, b: pd.DataFrame) -> None:
    n = len(b)
    if n == 0:
        print(f"{name}: no bets")
        return
    r = b["won"].mean()
    half = 1.96 * np.sqrt(r * (1 - r) / n)
    z = (r - BREAK_EVEN) / np.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / n)
    verdict = "PROMISING" if (r >= 0.55 and z >= 2) else "NOT SHOWN"
    print(f"{name}: {n:,} bets, won {r:.1%} (95% range {r - half:.1%} to {r + half:.1%}), "
          f"{z:+.1f} SE vs break-even -> {verdict}")
    for s, g in b.groupby("season"):
        print(f"    {s}: {len(g):,} bets, won {g['won'].mean():.1%}")


def main() -> None:
    d = pd.read_csv("vs_vegas_games.csv", parse_dates=["date"])
    d = d[d["dk_open"].notna() & d["dk_close"].notna()]
    seen = first_sample_ids()
    h = d[~d["game_id"].isin(seen)].copy()
    print(f"Holdout: {len(h):,} DraftKings games not in the first sample "
          f"({h['season'].value_counts().sort_index().to_dict()})\n")

    open_mkt = -h["dk_open"]
    h["edge"] = h["full_margin"] - open_mkt
    res = h["margin"] - open_mkt
    h = h[res != 0]
    h["won"] = np.sign(h["edge"]) == np.sign(res[h.index])

    judge("Rule A (key player out, edge 1+)",
          h[(h["miss_last_diff"].abs() >= 5) & (h["edge"].abs() >= 1)])
    judge("Rule B (edge 0-4)", h[h["edge"].abs() < 4])

    # Context only (not a betting rule): does the line still move toward the model?
    move = (-h["dk_close"]) - (-h["dk_open"])
    m = (move != 0) & (h["edge"].abs() >= 1)
    t = (np.sign(h.loc[m, "edge"]) == np.sign(move[m])).mean()
    print(f"\nContext: line moved toward the model in {t:.1%} of {m.sum():,} holdout games")


if __name__ == "__main__":
    main()
