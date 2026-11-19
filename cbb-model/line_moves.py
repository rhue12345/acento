"""Does the model know where the line is going?

    python line_moves.py     # after collect_lines.py and vs_vegas.py

Uses DraftKings games (2026-2027) with both an opening and closing spread.
"edge" = model's projected home margin minus the opening line's.
"move" = how far the market's expected home margin moved from open to close.

Tests, in order:
  1. How often the line moves toward the model (50% = no signal).
  2. The favorite check: lines may simply drift toward favorites and the
     model leans toward favorites. Split games by whether the model leans to
     the opening favorite or underdog; a real signal holds in both.
  3. Regression of the move on the model's edge, controlling for the
     opening line, with robust standard errors.
  4. Final margin regressed on the line and the model: does the model add
     information beyond the opening line? Beyond the closing line?
  5. Bets at the opening line on the model's side (break-even 52.4%).
Each is shown for both seasons separately, since a real effect should
show up in both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BREAK_EVEN = 110 / 210


def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> str:
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    cov = xtx_inv @ (X.T * resid**2) @ X @ xtx_inv  # HC0 robust
    se = np.sqrt(np.diag(cov))
    return "  ".join(f"{n} {b:+.3f} (t {b / s:+.1f})"
                     for n, b, s in zip(["const"] + names, beta, se))


def pct(mask_hits: pd.Series) -> str:
    n = len(mask_hits)
    if n == 0:
        return "   -"
    r = mask_hits.mean()
    return f"{r:6.1%} of {n:5,}  ({(r - 0.5) / np.sqrt(0.25 / n):+.1f} SE vs 50%)"


def main() -> None:
    d = pd.read_csv("vs_vegas_games.csv", parse_dates=["date"])
    d = d[d["dk_open"].notna() & d["dk_close"].notna()].copy()
    print(f"DraftKings games with open and close: {len(d):,} "
          f"({d['season'].value_counts().sort_index().to_dict()})")

    d["open_mkt"] = -d["dk_open"]         # market's expected home margin at open
    d["close_mkt"] = -d["dk_close"]
    d["move"] = d["close_mkt"] - d["open_mkt"]
    print(f"Line moved in {(d['move'] != 0).mean():.0%} of games; "
          f"average size when it moved {d.loc[d['move'] != 0, 'move'].abs().mean():.2f} pts\n")

    for model in ("baseline", "full"):
        d["edge"] = d[f"{model}_margin"] - d["open_mkt"]
        m = (d["move"] != 0) & (d["edge"].abs() >= 1)
        g = d[m]
        toward = np.sign(g["edge"]) == np.sign(g["move"])
        fav_home = g["open_mkt"] > 0
        lean_fav = np.sign(g["edge"]) == np.where(fav_home, 1, -1)
        to_fav = np.sign(g["move"]) == np.where(fav_home, 1, -1)

        print(f"===== {model} model =====")
        print("1. Line moved toward the model (model off the opener by 1+ pt):")
        print(f"   all          {pct(toward)}")
        for s in sorted(g["season"].unique()):
            print(f"   {s}         {pct(toward[g['season'] == s])}")
        print("2. Favorite check:")
        print(f"   base rate: line moved toward the opening favorite {pct(to_fav)}")
        print(f"   model leans to favorite: toward model {pct(toward[lean_fav])}")
        print(f"   model leans to underdog: toward model {pct(toward[~lean_fav])}")
        kp = g["miss_last_diff"].abs() >= 5
        print(f"   key player missed last game:   toward model {pct(toward[kp])}")
        print(f"   no key player missing:         toward model {pct(toward[~kp])}")

        print("3. Regression: move = a + b*edge + c*opening line")
        print("   " + ols(d["move"].to_numpy(), d[["edge", "open_mkt"]].to_numpy(),
                          ["edge", "open"]))
        for s in sorted(d["season"].unique()):
            x = d[d["season"] == s]
            print(f"   {s}: " + ols(x["move"].to_numpy(), x[["edge", "open_mkt"]].to_numpy(),
                                    ["edge", "open"]))

        print("4. Final margin = a + b*line + c*model  (c > 0 means the model adds information)")
        print("   vs opening line: " + ols(d["margin"].to_numpy(),
                                             d[["open_mkt", f"{model}_margin"]].to_numpy(),
                                             ["line", "model"]))
        print("   vs closing line: " + ols(d["margin"].to_numpy(),
                                             d[["close_mkt", f"{model}_margin"]].to_numpy(),
                                             ["line", "model"]))

        print("5. Betting the opening line on the model's side:")
        res = d["margin"] - d["open_mkt"]
        b = d[(res != 0)].assign(won=lambda x: np.sign(x["edge"]) == np.sign(res[x.index]))
        for lo, hi in [(0, 2), (2, 4), (4, 99)]:
            k = b[(b["edge"].abs() >= lo) & (b["edge"].abs() < hi)]
            r = k["won"].mean()
            se = np.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / len(k))
            print(f"   edge {lo}-{hi} pts: {len(k):5,} bets, won {r:.1%} "
                  f"({(r - BREAK_EVEN) / se:+.1f} SE vs break-even)")
        print()


if __name__ == "__main__":
    main()
