"""Point-in-time team strength features.

Every feature for a game is computed only from games played on earlier dates,
so the backtest never peeks at the result it is predicting.

Two rating systems run side by side:

* Elo with a margin-of-victory multiplier, carried across seasons with
  regression toward the mean.
* An adjusted scoring-margin rating (like KenPom/Massey/SRS): a ridge
  regression of each game's capped margin on team strengths plus home court,
  refit every game day, with games weighted toward the recent past and every
  team shrunk toward a preseason prior built from last season's rating.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

# --- Elo settings ---
ELO_MEAN = 1500.0
ELO_NEW_TEAM = 1250.0     # teams with no history (mostly non-D-I opponents)
ELO_K = 24.0
ELO_HOME = 90.0           # Elo points for a true home game
ELO_CARRYOVER = 0.65      # share of last season's deviation kept

# --- Adjusted margin settings ---
MARGIN_CAP = 25.0         # blowouts count as 25-point wins
PRIOR_CARRYOVER = 0.6     # preseason prior = 0.6 * last season's final rating
NEW_TEAM_PRIOR = -12.0    # prior for teams with no rating history
PRIOR_STRENGTH = 3.0      # prior is worth about this many games
RECENCY_HALF_LIFE = 60.0  # days; older games in a season count less


@dataclass
class RatingState:
    elo: dict[int, float] = field(default_factory=dict)
    margin_final: dict[int, float] = field(default_factory=dict)  # latest fit
    games_played: dict[int, int] = field(default_factory=dict)    # latest season


def _mov_multiplier(margin: float, elo_diff: float) -> float:
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff)


def _fit_margin_ratings(
    season_games: pd.DataFrame, as_of: pd.Timestamp, priors: dict[int, float]
) -> tuple[dict[int, float], float]:
    """Ridge-fit team ratings (points vs. average) from games before ``as_of``."""
    teams = sorted(priors)
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    prior_vec = np.array([priors[t] for t in teams])

    g = season_games
    n = len(g)
    if n == 0:
        return dict(zip(teams, prior_vec)), 3.0

    h = g["home_id"].map(idx).to_numpy()
    a = g["away_id"].map(idx).to_numpy()
    home = (~g["neutral"]).to_numpy().astype(float)
    margin = np.clip(
        (g["home_score"] - g["away_score"]).to_numpy(float), -MARGIN_CAP, MARGIN_CAP
    )
    age = (as_of - g["date"]).dt.days.to_numpy(float)
    w = np.sqrt(0.5 ** (age / RECENCY_HALF_LIFE))

    # Solve for deviations from the prior: margin - (prior_h - prior_a) =
    #   d_h - d_a + hca * home. The home column is scaled up so the ridge
    #   damping barely touches home court.
    hca_scale = 10.0
    rows = np.repeat(np.arange(n), 3)
    cols = np.column_stack([h, a, np.full(n, n_teams)]).ravel()
    vals = np.column_stack([w, -w, w * home / hca_scale]).ravel()
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n_teams + 1))
    y = w * (margin - (prior_vec[h] - prior_vec[a]))
    sol = lsqr(X, y, damp=np.sqrt(PRIOR_STRENGTH), atol=1e-8, btol=1e-8)[0]

    ratings = prior_vec + sol[:n_teams]
    hca = sol[n_teams] / hca_scale
    return dict(zip(teams, ratings)), hca


def build_features(games: pd.DataFrame) -> tuple[pd.DataFrame, RatingState]:
    """Walk through all games in date order and attach pre-game features.

    Returns the games with feature columns (completed and scheduled), and the
    rating state after the last completed game.
    """
    state = RatingState()
    out = []

    for season, sgames in games.groupby("season", sort=True):
        # --- new season: regress Elo, set margin priors ---
        for t in list(state.elo):
            state.elo[t] = ELO_MEAN + ELO_CARRYOVER * (state.elo[t] - ELO_MEAN)
        season_teams = set(sgames["home_id"]) | set(sgames["away_id"])
        priors = {
            t: PRIOR_CARRYOVER * state.margin_final[t]
            if t in state.margin_final
            else NEW_TEAM_PRIOR
            for t in season_teams
        }
        games_played = {t: 0 for t in season_teams}

        done = sgames[sgames["completed"]]
        played: list[pd.DataFrame] = []
        ratings, hca = dict(priors), 3.0

        for date, day in sgames.groupby(sgames["date"].dt.normalize(), sort=True):
            # Refit margin ratings on everything before this day.
            prior_games = done[done["date"].dt.normalize() < date]
            if len(prior_games) and (not played or len(prior_games) != len(played[-1])):
                ratings, hca = _fit_margin_ratings(prior_games, date, priors)
                played.append(prior_games)

            rows = day.copy()
            rows["home_elo"] = [state.elo.get(t, ELO_NEW_TEAM) for t in day["home_id"]]
            rows["away_elo"] = [state.elo.get(t, ELO_NEW_TEAM) for t in day["away_id"]]
            rows["home_rating"] = day["home_id"].map(ratings).to_numpy()
            rows["away_rating"] = day["away_id"].map(ratings).to_numpy()
            rows["home_gp"] = day["home_id"].map(games_played).to_numpy()
            rows["away_gp"] = day["away_id"].map(games_played).to_numpy()
            rows["hca_est"] = hca
            out.append(rows)

            # Update Elo and games played with this day's results.
            for g in day[day["completed"]].itertuples():
                eh = state.elo.get(g.home_id, ELO_NEW_TEAM)
                ea = state.elo.get(g.away_id, ELO_NEW_TEAM)
                diff = eh - ea + (0.0 if g.neutral else ELO_HOME)
                expected = 1.0 / (1.0 + 10 ** (-diff / 400.0))
                margin = g.home_score - g.away_score
                result = 1.0 if margin > 0 else 0.0
                winner_diff = diff if margin > 0 else -diff
                delta = ELO_K * _mov_multiplier(margin, winner_diff) * (result - expected)
                state.elo[g.home_id] = eh + delta
                state.elo[g.away_id] = ea - delta
                games_played[g.home_id] += 1
                games_played[g.away_id] += 1

        # End of season: final margin ratings become next season's priors.
        if len(done):
            end = done["date"].max().normalize() + pd.Timedelta(days=1)
            final, _ = _fit_margin_ratings(done, end, priors)
            state.margin_final.update(final)
        state.games_played = games_played

    feats = pd.concat(out, ignore_index=True)
    return add_model_inputs(feats), state


def add_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw ratings into home-perspective model inputs."""
    df = df.copy()
    df["home"] = (~df["neutral"]).astype(float)
    df["elo_diff"] = (df["home_elo"] - df["away_elo"]) / 100.0
    df["rating_diff"] = df["home_rating"] - df["away_rating"]
    # Early in the season, current-season evidence is thin; let the model
    # learn how much to trust each rating as games accumulate.
    early = np.exp(-np.minimum(df["home_gp"], df["away_gp"]) / 5.0)
    df["elo_diff_early"] = df["elo_diff"] * early
    df["rating_diff_early"] = df["rating_diff"] * early
    if "home_score" in df:
        df["margin"] = df["home_score"] - df["away_score"]
        df["home_win"] = (df["margin"] > 0).astype(int)
    return df


FEATURES = ["home", "elo_diff", "rating_diff", "elo_diff_early", "rating_diff_early"]
