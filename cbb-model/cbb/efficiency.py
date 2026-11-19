"""Possession-based ratings and situational features.

Points per 100 possessions strips out tempo: a 70-68 grind and a 90-88
shootout carry the same information about which team is better. Each team
gets an offensive rating, a defensive rating and a tempo rating, refit every
game day from earlier games only (same walk-forward rule as ``ratings.py``).

Also adds rest: days since each team's previous game.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.sparse.linalg import lsqr

from cbb.data import DATA_DIR

BOX_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/"
    "mbb/team_box/parquet/team_box_{season}.parquet"
)


@dataclass(frozen=True)
class EffParams:
    half_life: float = 60.0      # days; recency weighting within a season
    prior_games: float = 4.0     # strength of the preseason prior, in games
    carryover: float = 0.6       # share of last season's rating kept
    new_team: float = -8.0       # off/def prior for teams with no history


def load_box(seasons: list[int], refresh_latest: bool = True) -> pd.DataFrame:
    """Possessions and points for each team in each game."""
    frames = []
    for season in seasons:
        path = DATA_DIR / f"team_box_{season}.parquet"
        if not path.exists() or (refresh_latest and season == max(seasons)):
            for attempt in range(6):
                r = requests.get(BOX_URL.format(season=season), timeout=60)
                if r.status_code != 429:
                    break
                time.sleep(5 * 2**attempt)
            r.raise_for_status()
            path.write_bytes(r.content)
        frames.append(pd.read_parquet(path, columns=[
            "game_id", "team_id", "team_score", "field_goals_attempted",
            "offensive_rebounds", "total_turnovers", "free_throws_attempted"]))
    box = pd.concat(frames, ignore_index=True).dropna()
    box["team_id"] = box["team_id"].astype(int)
    box["poss"] = (box["field_goals_attempted"] - box["offensive_rebounds"]
                   + box["total_turnovers"] + 0.475 * box["free_throws_attempted"])
    box = box[box["poss"] > 40]  # drop broken box scores
    return box[["game_id", "team_id", "team_score", "poss"]].drop_duplicates(
        ["game_id", "team_id"])


def _attach_box(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Add per-game tempo and each side's points per 100 possessions."""
    g = games.merge(box.rename(columns={"team_id": "home_id", "poss": "h_poss"})
                    [["game_id", "home_id", "h_poss"]], on=["game_id", "home_id"], how="left")
    g = g.merge(box.rename(columns={"team_id": "away_id", "poss": "a_poss"})
                [["game_id", "away_id", "a_poss"]], on=["game_id", "away_id"], how="left")
    g["pace"] = (g["h_poss"] + g["a_poss"]) / 2
    g["h_eff"] = 100 * g["home_score"] / g["pace"]
    g["a_eff"] = 100 * g["away_score"] / g["pace"]
    return g


def _fit(g: pd.DataFrame, as_of: pd.Timestamp, teams: list[int], prior_o, prior_d,
         prior_t, p: EffParams):
    """Ridge fit of offense, defense, tempo deviations from the priors."""
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    h = g["home_id"].map(idx).to_numpy()
    a = g["away_id"].map(idx).to_numpy()
    home = (~g["neutral"]).to_numpy(float)
    w = np.sqrt(0.5 ** ((as_of - g["date"]).dt.days.to_numpy(float) / p.half_life))
    n = len(g)
    mu = np.mean(np.r_[g["h_eff"], g["a_eff"]])
    tmu = g["pace"].mean()
    scale = 10.0  # keeps the ridge penalty off home court

    # Efficiency: columns = offense (n_t), defense (n_t), home court.
    # home eff = mu + o_h - d_a + hca ; away eff = mu + o_a - d_h - hca
    r = np.arange(2 * n)
    rows = np.repeat(r, 3)
    oc = np.r_[h, a]
    dc = n_t + np.r_[a, h]
    hc = np.full(2 * n, 2 * n_t)
    ww = np.r_[w, w]
    sign = np.r_[home, -home]
    X = sparse.csr_matrix((np.column_stack([ww, -ww, ww * sign / scale]).ravel(),
                           (rows, np.column_stack([oc, dc, hc]).ravel())),
                          shape=(2 * n, 2 * n_t + 1))
    y = np.r_[g["h_eff"], g["a_eff"]] - mu - (prior_o[oc] - prior_d[dc - n_t])
    sol = lsqr(X, ww * y, damp=np.sqrt(p.prior_games), atol=1e-8, btol=1e-8)[0]
    off = prior_o + sol[:n_t]
    dfn = prior_d + sol[n_t:2 * n_t]
    hca = sol[2 * n_t] / scale

    # Tempo: pace = tmu + t_h + t_a
    Xt = sparse.csr_matrix((np.column_stack([w, w]).ravel(),
                            (np.repeat(np.arange(n), 2), np.column_stack([h, a]).ravel())),
                           shape=(n, n_t))
    yt = g["pace"].to_numpy() - tmu - (prior_t[h] + prior_t[a])
    tsol = lsqr(Xt, w * yt, damp=np.sqrt(p.prior_games), atol=1e-8, btol=1e-8)[0]
    tempo = prior_t + tsol
    return off, dfn, tempo, mu, tmu, hca


def build_eff_features(games: pd.DataFrame, box: pd.DataFrame,
                       p: EffParams = EffParams()) -> pd.DataFrame:
    """Pre-game efficiency features for every game, keyed by game_id."""
    games = _attach_box(games, box)
    final_o, final_d, final_t = {}, {}, {}
    out = []

    for season, sg in games.groupby("season", sort=True):
        teams = sorted(set(sg["home_id"]) | set(sg["away_id"]))
        prior_o = np.array([p.carryover * final_o.get(t, p.new_team / p.carryover) for t in teams])
        prior_d = np.array([p.carryover * final_d.get(t, p.new_team / p.carryover) for t in teams])
        prior_t = np.array([p.carryover * final_t.get(t, 0.0) for t in teams])
        idx = {t: i for i, t in enumerate(teams)}
        fit_on = sg[sg["completed"] & sg["pace"].notna()]
        state = (prior_o, prior_d, prior_t, 100.0, 68.0, 3.0)
        last_n = 0

        for date, day in sg.groupby(sg["date"].dt.normalize(), sort=True):
            prev = fit_on[fit_on["date"].dt.normalize() < date]
            if len(prev) != last_n and len(prev) > 0:
                state = _fit(prev, date, teams, prior_o, prior_d, prior_t, p)
                last_n = len(prev)
            off, dfn, tempo, mu, tmu, hca = state
            h = day["home_id"].map(idx).to_numpy()
            a = day["away_id"].map(idx).to_numpy()
            home = (~day["neutral"]).to_numpy(float)
            pace = tmu + tempo[h] + tempo[a]
            h_pts = (mu + off[h] - dfn[a] + hca * home) * pace / 100
            a_pts = (mu + off[a] - dfn[h] - hca * home) * pace / 100
            out.append(pd.DataFrame({
                "game_id": day["game_id"].to_numpy(),
                "eff_margin": h_pts - a_pts,
                "eff_off_diff": off[h] - off[a],
                "eff_def_diff": dfn[h] - dfn[a],
                "pace_pred": pace,
            }))

        if len(fit_on):
            end = fit_on["date"].max().normalize() + pd.Timedelta(days=1)
            off, dfn, tempo, *_ = _fit(fit_on, end, teams, prior_o, prior_d, prior_t, p)
            for t, i in idx.items():
                final_o[t], final_d[t], final_t[t] = off[i], dfn[i], tempo[i]

    return pd.concat(out, ignore_index=True)


def rest_features(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game (capped at 7; season opener = 7)."""
    long = pd.concat([
        games[["game_id", "season", "date", "home_id"]].rename(columns={"home_id": "team"}).assign(side="home"),
        games[["game_id", "season", "date", "away_id"]].rename(columns={"away_id": "team"}).assign(side="away"),
    ]).sort_values("date")
    day = long["date"].dt.normalize()
    long["rest"] = (day - day.groupby([long["season"], long["team"]]).shift()).dt.days
    long["rest"] = long["rest"].fillna(7).clip(upper=7)
    wide = long.pivot_table(index="game_id", columns="side", values="rest", aggfunc="first")
    return pd.DataFrame({"game_id": wide.index,
                         "rest_diff": (wide["home"] - wide["away"]).to_numpy(),
                         "home_b2b": (wide["home"] <= 1).astype(float).to_numpy(),
                         "away_b2b": (wide["away"] <= 1).astype(float).to_numpy()})
