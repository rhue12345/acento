"""Predict upcoming games, or any matchup.

    python predict.py                        # games in the next 3 days
    python predict.py --days 7
    python predict.py --matchup "Duke" "Kansas" --neutral
    python predict.py --top 25               # current power rankings

Re-downloads the current season first, so ratings include last night's games.
"""
from __future__ import annotations

import argparse
import pickle

import pandas as pd

from cbb.data import load_games
from cbb.ratings import ELO_NEW_TEAM, NEW_TEAM_PRIOR, add_model_inputs, build_features
from train import MODEL_PATH, current_season, predict_proba

HISTORY_SEASONS = 4  # enough for the season-to-season priors to settle


def load_model() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def current_ratings(feats: pd.DataFrame, state, season: int) -> pd.DataFrame:
    """Ratings after the latest completed game for every team in ``season``."""
    s = feats[feats["season"] == season]
    names = dict(zip(s["home_id"], s["home_team"])) | dict(zip(s["away_id"], s["away_team"]))
    df = pd.DataFrame({"team_id": list(names), "team": list(names.values())})
    df["rating"] = df["team_id"].map(state.margin_final).fillna(NEW_TEAM_PRIOR)
    df["elo"] = df["team_id"].map(state.elo).fillna(ELO_NEW_TEAM)
    df["games"] = df["team_id"].map(state.games_played).fillna(0).astype(int)
    return df.sort_values("rating", ascending=False).reset_index(drop=True)


def find_team(ratings: pd.DataFrame, query: str) -> pd.Series:
    q = query.lower()
    exact = ratings[ratings["team"].str.lower() == q]
    if len(exact):
        return exact.iloc[0]
    hits = ratings[ratings["team"].str.lower().str.contains(q, regex=False)]
    if hits.empty:
        raise SystemExit(f"No team matches '{query}'")
    if len(hits) > 1:
        print(f"'{query}' matched {len(hits)} teams; using {hits.iloc[0]['team']} "
              f"(others: {', '.join(hits['team'].iloc[1:6])})")
    return hits.iloc[0]


def matchup_frame(home: pd.Series, away: pd.Series, neutral: bool) -> pd.DataFrame:
    return add_model_inputs(pd.DataFrame([{
        "neutral": neutral,
        "home_elo": home["elo"], "away_elo": away["elo"],
        "home_rating": home["rating"], "away_rating": away["rating"],
        "home_gp": home["games"], "away_gp": away["games"],
    }]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--matchup", nargs=2, metavar=("HOME", "AWAY"))
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--top", type=int)
    args = ap.parse_args()

    bundle = load_model()
    models = bundle["models"]
    season = current_season()
    games = load_games(list(range(season - HISTORY_SEASONS, season + 1)))
    feats, state = build_features(games)
    ratings = current_ratings(feats, state, season)

    if args.top:
        print(f"Top {args.top} by adjusted margin (points better than an average team):")
        print(ratings.head(args.top)[["team", "rating", "elo", "games"]]
              .round(1).to_string(index=True))
        return

    if args.matchup:
        home, away = (find_team(ratings, q) for q in args.matchup)
        X = matchup_frame(home, away, args.neutral)
        p = predict_proba(models, X)["ensemble"][0]
        spread = models["spread"].predict(X[bundle["features"]])[0]
        site = "neutral court" if args.neutral else f"at {home['team']}"
        fav, fp = (home["team"], p) if p >= 0.5 else (away["team"], 1 - p)
        print(f"{away['team']} vs {home['team']} ({site})")
        print(f"  {fav} wins with probability {fp:.1%}; "
              f"projected margin {abs(spread):.1f} points")
        return

    today = pd.Timestamp.today().normalize()
    upcoming = feats[(~feats["completed"]) & (feats["season"] == season)
                     & (feats["date"] >= today)
                     & (feats["date"] < today + pd.Timedelta(days=args.days))].copy()
    if upcoming.empty:
        print("No scheduled games in that window.")
        return
    # Feature rows were built before each game day; refresh them with the
    # newest ratings so games a few days out use last night's results too.
    by_id = ratings.set_index("team_id")
    for side in ("home", "away"):
        ids = upcoming[f"{side}_id"]
        upcoming[f"{side}_rating"] = ids.map(by_id["rating"]).fillna(NEW_TEAM_PRIOR).to_numpy()
        upcoming[f"{side}_elo"] = ids.map(by_id["elo"]).fillna(ELO_NEW_TEAM).to_numpy()
        upcoming[f"{side}_gp"] = ids.map(by_id["games"]).fillna(0).to_numpy()
    upcoming = add_model_inputs(upcoming.drop(columns=["home_score", "away_score"]))
    upcoming["p_home"] = predict_proba(models, upcoming)["ensemble"]
    upcoming["spread"] = models["spread"].predict(upcoming[bundle["features"]])

    for r in upcoming.sort_values("date").itertuples():
        fav, dog, p = ((r.home_team, r.away_team, r.p_home) if r.p_home >= 0.5
                       else (r.away_team, r.home_team, 1 - r.p_home))
        sep = "vs" if r.neutral else "at"
        game = f"{r.away_team} {sep} {r.home_team}"
        print(f"{r.date:%a %b %d}  {game:62s}  {fav:30s} {p:6.1%}  by {abs(r.spread):4.1f}")


if __name__ == "__main__":
    main()
