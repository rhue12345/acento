"""Download and tidy men's college basketball game results.

Source: the sportsdataverse hoopR-mbb-data repository on GitHub, which mirrors
ESPN's men's college basketball schedules and scores (2006 onward, updated
daily during the season). Files are cached under ``data/``.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/"
    "mbb/schedules/parquet/mbb_schedule_{season}.parquet"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLUMNS = {
    "game_id": "game_id",
    "season": "season",
    "season_type": "season_type",
    "game_date": "date",
    "neutral_site": "neutral",
    "status_type_completed": "completed",
    "status_type_name": "status",
    "home_id": "home_id",
    "home_display_name": "home_team",
    "away_id": "away_id",
    "away_display_name": "away_team",
    "home_score": "home_score",
    "away_score": "away_score",
}


def download_season(season: int, refresh: bool = False) -> Path:
    """Fetch one season's schedule file (season 2027 = the 2026-27 season)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"mbb_schedule_{season}.parquet"
    if path.exists() and not refresh:
        return path
    for attempt in range(6):
        resp = requests.get(BASE_URL.format(season=season), timeout=60)
        if resp.status_code != 429:
            break
        time.sleep(5 * 2**attempt)  # GitHub rate limit: back off and retry
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def load_games(seasons: list[int], refresh_latest: bool = True) -> pd.DataFrame:
    """Return one row per game (completed and scheduled) across ``seasons``.

    The most recent season is re-downloaded when ``refresh_latest`` is set,
    so new results and schedule changes are picked up.
    """
    frames = []
    latest = max(seasons)
    for season in seasons:
        path = download_season(season, refresh=refresh_latest and season == latest)
        df = pd.read_parquet(path, columns=list(COLUMNS))
        frames.append(df.rename(columns=COLUMNS))
    games = pd.concat(frames, ignore_index=True)

    games["date"] = pd.to_datetime(games["date"])
    games["neutral"] = games["neutral"].fillna(False).astype(bool)
    games["completed"] = games["completed"].fillna(False).astype(bool)
    games["home_id"] = games["home_id"].astype(int)
    games["away_id"] = games["away_id"].astype(int)
    games = games[games["status"].isin(["STATUS_FINAL", "STATUS_SCHEDULED"])]
    # Drop "completed" games with no real score (forfeits, data glitches).
    bad = games["completed"] & (games["home_score"] + games["away_score"] <= 0)
    games = games[~bad]
    games = games.drop_duplicates("game_id").sort_values(["date", "game_id"])
    return games.reset_index(drop=True)
