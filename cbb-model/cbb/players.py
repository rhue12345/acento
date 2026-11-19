"""Who's playing: player-availability features from player box scores.

Each player's value is his average Game Score (John Hollinger's one-number
box-score summary, roughly "points of production") over the team's earlier
games this season. A "key player" averages 20+ minutes and has played in at
least 70% of the team's earlier games (3+ games). Bench players who sit by the
coach's choice are left out, since their absences are routine noise.

Two kinds of feature per team and game:

* ``missing_last``: total value of key players who sat out the team's previous
  game. Known before tip-off, so it is a legitimate predictor (a player who
  missed the last game is often still out).
* ``missing_now``: total value of key players who did not play in *this* game.
  Only known from the box score afterward. It stands in for a perfect injury
  report and measures how much that information would be worth; it must never
  be used for real predictions.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from cbb.data import DATA_DIR

URL = ("https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/"
       "mbb/player_box/parquet/player_box_{season}.parquet")
COLS = ["game_id", "season", "game_date", "athlete_id", "team_id", "minutes",
        "points", "field_goals_made", "field_goals_attempted", "free_throws_made",
        "free_throws_attempted", "offensive_rebounds", "defensive_rebounds",
        "assists", "steals", "blocks", "turnovers", "fouls", "did_not_play"]


def load_player_box(seasons: list[int], refresh_latest: bool = True) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = DATA_DIR / f"player_box_{season}.parquet"
        if not path.exists() or (refresh_latest and season == max(seasons)):
            for attempt in range(6):
                r = requests.get(URL.format(season=season), timeout=120)
                if r.status_code != 429:
                    break
                time.sleep(5 * 2**attempt)
            r.raise_for_status()
            path.write_bytes(r.content)
        frames.append(pd.read_parquet(path, columns=COLS))
    pb = pd.concat(frames, ignore_index=True)
    pb = pb.dropna(subset=["athlete_id", "team_id"])
    pb["athlete_id"] = pb["athlete_id"].astype(int)
    pb["team_id"] = pb["team_id"].astype(int)
    stats = pb[COLS[6:-1]].fillna(0)
    pb["game_score"] = (stats["points"] + 0.4 * stats["field_goals_made"]
                        - 0.7 * stats["field_goals_attempted"]
                        - 0.4 * (stats["free_throws_attempted"] - stats["free_throws_made"])
                        + 0.7 * stats["offensive_rebounds"] + 0.3 * stats["defensive_rebounds"]
                        + stats["steals"] + 0.7 * stats["assists"] + 0.7 * stats["blocks"]
                        - 0.4 * stats["fouls"] - stats["turnovers"])
    pb["played"] = (~pb["did_not_play"].fillna(False).astype(bool)) & (pb["minutes"].fillna(0) > 0)
    pb["minutes"] = pb["minutes"].fillna(0)
    return pb[["game_id", "season", "game_date", "athlete_id", "team_id", "played",
               "game_score", "minutes"]]


def availability_features(pb: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team_id) with missing_last and missing_now."""
    # Games each team played, in order (from any row of that team's box).
    tg = (pb[["season", "team_id", "game_id", "game_date"]].drop_duplicates(["team_id", "game_id"])
          .sort_values(["season", "team_id", "game_date", "game_id"]))
    played = pb[pb["played"]]
    played_by_tg = played.groupby(["team_id", "game_id"])
    roster_gs = {k: dict(zip(v["athlete_id"], zip(v["game_score"], v["minutes"])))
                 for k, v in played_by_tg}

    rows = []
    for (season, team), games in tg.groupby(["season", "team_id"], sort=False):
        n_games = 0
        gp = defaultdict(int)        # games played per player so far
        gs_sum = defaultdict(float)  # total game score so far
        min_sum = defaultdict(float)  # total minutes so far
        last_played: set[int] = set()
        for gid in games["game_id"]:
            regulars = {p: gs_sum[p] / gp[p] for p in gp
                        if n_games >= 3 and gp[p] >= 3 and gp[p] >= 0.7 * n_games
                        and min_sum[p] / gp[p] >= 20}
            now = roster_gs.get((team, gid), {})
            missing_last = sum(v for p, v in regulars.items() if p not in last_played) if n_games else 0.0
            missing_now = sum(v for p, v in regulars.items() if p not in now)
            rows.append((gid, team, missing_last, missing_now, sum(regulars.values())))
            for p, (g, mins) in now.items():
                gp[p] += 1
                gs_sum[p] += g
                min_sum[p] += mins
            n_games += 1
            last_played = set(now)
    return pd.DataFrame(rows, columns=["game_id", "team_id", "missing_last", "missing_now",
                                       "regulars_value"])


def game_level(av: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away availability features keyed by game_id."""
    g = games[["game_id", "home_id", "away_id"]]
    h = av.rename(columns={"team_id": "home_id"}).add_prefix("h_").rename(
        columns={"h_game_id": "game_id", "h_home_id": "home_id"})
    a = av.rename(columns={"team_id": "away_id"}).add_prefix("a_").rename(
        columns={"a_game_id": "game_id", "a_away_id": "away_id"})
    g = g.merge(h, on=["game_id", "home_id"], how="left").merge(a, on=["game_id", "away_id"], how="left")
    g = g.fillna(0.0)
    return pd.DataFrame({
        "game_id": g["game_id"],
        # Positive = the home team is missing more production than the away team.
        "miss_last_diff": g["h_missing_last"] - g["a_missing_last"],
        "miss_now_diff": g["h_missing_now"] - g["a_missing_now"],
    })
