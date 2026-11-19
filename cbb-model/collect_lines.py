"""Collect historical betting lines from ESPN game files.

    python collect_lines.py                 # 1,000 sampled D-I games per season
    python collect_lines.py --per-season 0  # every D-I game (slow)

Source: sportsdataverse/hoopR-mbb-raw, which stores ESPN's game JSON. Its
"pickcenter" block holds the lines (consensus/Caesars through 2023, DraftKings
with open and close from 2026; 2024-2025 have none). Resumable: games already
in data/lines_raw.jsonl are skipped. Only the line data is kept, not the 0.5-1 MB
game files.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from cbb.data import DATA_DIR

URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-raw/main/mbb/json/final/{}.json"
OUT = DATA_DIR / "lines_raw.jsonl"
SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2026, 2027]
DROP = {"links", "link", "header", "team", "logos"}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def fetch(game_id: int) -> dict | None:
    for attempt in range(7):
        try:
            r = requests.get(URL.format(game_id), timeout=90)
        except requests.RequestException:
            time.sleep(5 * 2**attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * 2**attempt)
            continue
        if r.status_code != 200:
            return {"game_id": int(game_id), "missing": True}
        d = r.json()
        comp = d.get("header", {}).get("competitions", [{}])[0]
        home = next((c["team"]["id"] for c in comp.get("competitors", [])
                     if c.get("homeAway") == "home"), None)
        return {"game_id": int(game_id), "home_team_id": home,
                "pickcenter": strip(d.get("pickcenter") or [])}
    return None  # gave up; will be retried next run


def d1_games(season: int) -> pd.DataFrame:
    s = pd.read_parquet(DATA_DIR / f"mbb_schedule_{season}.parquet",
                        columns=["game_id", "home_id", "away_id", "status_type_completed"])
    ref = s if season != 2027 else pd.read_parquet(
        DATA_DIR / "mbb_schedule_2026.parquet", columns=["home_id", "away_id"])
    counts = pd.concat([ref["home_id"], ref["away_id"]]).value_counts()
    d1 = set(counts[counts >= 20].index)
    s = s[s["status_type_completed"] & s["home_id"].isin(d1) & s["away_id"].isin(d1)]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-season", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    done = set()
    if OUT.exists():
        done = {json.loads(line)["game_id"] for line in OUT.open()}
    todo = []
    for season in SEASONS:
        g = d1_games(season)
        if args.per_season and len(g) > args.per_season:
            g = g.sample(args.per_season, random_state=season)
        todo += [x for x in g["game_id"] if int(x) not in done]
    print(f"{len(done):,} games already collected, {len(todo):,} to fetch", flush=True)

    with OUT.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        for i, rec in enumerate(pool.map(fetch, todo), 1):
            if rec is not None:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if i % 250 == 0:
                print(f"  {i:,}/{len(todo):,}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
