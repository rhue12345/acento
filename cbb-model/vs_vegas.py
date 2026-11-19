"""Score the model against the betting market, against the spread (ATS).

    python vs_vegas.py      # after collect_lines.py

For every game with a line: the model's projected margin vs the market's.
The model "bets" the side it likes; a bet wins if that side covers. At
standard -110 odds you need 52.4% to break even. Pushes are excluded.

Line used:
  2026-2027: DraftKings closing line (opening line also kept)
  2016-2023: TeamRankings line (has odds; timing not documented), consensus
             as fallback. numberFire is a projection site, not a book; it is
             scored separately as "another model".

Models are walk-forward (each season predicted from earlier seasons only).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cbb.data import DATA_DIR, load_games
from cbb.efficiency import EffParams, build_eff_features, load_box, rest_features
from cbb.players import availability_features, game_level, load_player_box
from cbb.ratings import FEATURES as BASE
from tossups import NEW, add_new_features, d1_mask, walk_forward

BREAK_EVEN = 110 / 210
MODELS = {"baseline": BASE, "full": NEW + ["miss_last_diff"]}


def num(x):
    try:
        return float(str(x).lstrip("+ou"))
    except (TypeError, ValueError):
        return np.nan


def parse_lines() -> pd.DataFrame:
    rows = []
    for line in (DATA_DIR / "lines_raw.jsonl").open():
        r = json.loads(line)
        row = {"game_id": r["game_id"]}
        for p in r.get("pickcenter") or []:
            name = (p.get("provider") or {}).get("name", "")
            if p.get("spread") is None:
                continue
            if name == "DraftKings":
                ps = p.get("pointSpread", {}).get("home", {})
                row["dk_close"] = num(ps.get("close", {}).get("line"))
                row["dk_open"] = num(ps.get("open", {}).get("line"))
                if np.isnan(row["dk_close"]):
                    row["dk_close"] = p["spread"]
            elif name in ("teamrankings", "consensus", "numberfire"):
                row[name] = p["spread"]
            elif "Caesars" in name:
                row["caesars"] = p["spread"]
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ["dk_close", "dk_open", "teamrankings", "consensus", "numberfire", "caesars"]:
        if c not in df:
            df[c] = np.nan
    # Spreads are quoted for the home team: -6.5 = home favored by 6.5.
    df["line"] = df["dk_close"].fillna(df["teamrankings"]).fillna(df["caesars"]).fillna(df["consensus"])
    df["source"] = np.select([df["dk_close"].notna(), df["teamrankings"].notna(),
                              df["caesars"].notna(), df["consensus"].notna()],
                             ["DraftKings close", "TeamRankings", "Caesars", "consensus"], "none")
    return df[df["line"].notna()]


def ats(d: pd.DataFrame, model_margin: str, line: str = "line") -> pd.DataFrame:
    """Bet the model's side vs ``line``; return rows with edge and result."""
    market = -d[line]                      # market's expected home margin
    edge = d[model_margin] - market        # + = model likes home more than market
    result = d["margin"] - market          # + = home covered
    out = d.assign(edge=edge, abs_edge=edge.abs(), result=result)
    out = out[(out["result"] != 0) & (out["edge"] != 0)]
    out["won"] = np.sign(out["edge"]) == np.sign(out["result"])
    return out


def report(name: str, b: pd.DataFrame) -> None:
    n = len(b)
    if n == 0:
        return
    rate = b["won"].mean()
    se = np.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / n)
    z = (rate - BREAK_EVEN) / se
    print(f"  {name:38s} {n:6,} bets  {rate:6.1%}  ({z:+.1f} SE vs break-even)")


def main() -> None:
    lines = parse_lines()
    print(f"{len(lines):,} games with a line:", lines["source"].value_counts().to_dict())

    seasons = list(range(2008, 2028))
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    games = load_games(seasons, refresh_latest=False)
    done = feats[feats["completed"] & (feats["season"] > 2008)].copy()
    done = done[d1_mask(done) | done["game_id"].isin(lines["game_id"])]
    eff = build_eff_features(games, load_box(seasons, refresh_latest=False),
                             EffParams(half_life=120.0, prior_games=2.0))
    av = game_level(availability_features(load_player_box(seasons, refresh_latest=False)), games)
    df = add_new_features(done, eff, rest_features(games)).merge(av, on="game_id", how="left")
    df[["miss_last_diff", "miss_now_diff"]] = df[["miss_last_diff", "miss_now_diff"]].fillna(0)

    test_seasons = sorted(set(df.loc[df["game_id"].isin(lines["game_id"]), "season"]))
    preds = None
    for name, f in MODELS.items():
        wf = walk_forward(df, f, test_seasons)[["game_id", "spread", "p"]]
        wf = wf.rename(columns={"spread": f"{name}_margin", "p": f"{name}_p"})
        preds = wf if preds is None else preds.merge(wf, on="game_id")
    d = (df[["game_id", "season", "date", "home_team", "away_team", "margin", "neutral",
             "home_rating", "away_rating", "miss_last_diff", "home_gp", "away_gp"]]
         .merge(preds, on="game_id").merge(lines, on="game_id"))
    print(f"{len(d):,} games with both a line and a model prediction\n")

    # How close is each model to the market at all?
    for name in MODELS:
        mae = np.mean(np.abs(d[f"{name}_margin"] - d["margin"]))
        mkt = np.mean(np.abs(-d["line"] - d["margin"]))
        print(f"Avg miss on final margin: {name} model {mae:.2f} pts, market {mkt:.2f} pts")
    nf = d[d["numberfire"].notna()]
    if len(nf):
        print(f"  (numberFire projections on their {len(nf):,} games: "
              f"{np.mean(np.abs(-nf['numberfire'] - nf['margin'])):.2f} pts)")

    for name in MODELS:
        b = ats(d, f"{name}_margin")
        print(f"\n=== {name} model vs the line (break-even {BREAK_EVEN:.1%}) ===")
        report("every game", b)
        for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]:
            report(f"model disagrees by {lo}-{hi} pts", b[(b["abs_edge"] >= lo) & (b["abs_edge"] < hi)])
        big = b[b["abs_edge"] >= 3]
        print("  -- where the model disagrees by 3+ points --")
        report("November-December", big[big["date"].dt.month.isin([11, 12])])
        report("January onward", big[~big["date"].dt.month.isin([11, 12])])
        avg = (big["home_rating"] + big["away_rating"]) / 2
        report("lower-rated matchups (avg rating < 0)", big[avg < 0])
        report("higher-rated matchups (avg rating >= 0)", big[avg >= 0])
        report("a key player missed last game", big[big["miss_last_diff"].abs() >= 5])
        print("  -- by line source --")
        for src, g in b.groupby("source"):
            report(src, g)
        print("  -- by season --")
        for s, g in b.groupby("season"):
            report(str(s), g)

    # Closing line value: does the line move toward the model between open and close?
    dk = d[d["dk_open"].notna() & d["dk_close"].notna()].copy()
    if len(dk):
        print(f"\n=== Line movement, DraftKings 2026-2027 ({len(dk):,} games) ===")
        print("If the model knows something, the line should move toward it before tip-off.")
        for name in MODELS:
            edge = dk[f"{name}_margin"] - (-dk["dk_open"])
            move = (-dk["dk_close"]) - (-dk["dk_open"])
            m = (move != 0) & (edge.abs() >= 1)
            toward = (np.sign(edge[m]) == np.sign(move[m])).mean()
            print(f"  {name}: line moved toward the model in {toward:.1%} of {m.sum():,} "
                  f"games where it moved and the model disagreed by 1+ pt (50% = no signal)")
            b = ats(dk, f"{name}_margin", line="dk_open")
            report(f"{name}: betting the OPENING line", b)
            report(f"{name}: betting the OPENING line, edge 3+", b[b["abs_edge"] >= 3])

    d.to_csv("vs_vegas_games.csv", index=False)
    print("\nPer-game detail: vs_vegas_games.csv")


if __name__ == "__main__":
    main()
