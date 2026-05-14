"""
generate_data.py
----------------
Generates a realistic synthetic NBA playoff game dataset based on
real NBA statistics distributions (2014-2024 era).

Run this once to produce data/playoff_games.csv.
To use real data instead, see README.md → "Swapping in Live Data".
"""

import numpy as np
import pandas as pd
import os

SEED = 42
rng = np.random.default_rng(SEED)

# ── Real NBA team skill tiers (approximate 2014-2024 era) ──────────────────
TEAM_PROFILES = {
    # format: (off_rtg_mean, def_rtg_mean, pace_mean)
    # Elite offenses
    "GSW": (115.0, 108.5, 100.5),
    "BOS": (113.5, 108.0,  97.0),
    "MIL": (114.0, 109.0,  99.5),
    "PHX": (113.0, 110.0,  99.0),
    "DEN": (114.5, 111.0,  97.5),
    "MIA": (110.5, 108.5,  95.5),
    # Mid-tier contenders
    "LAL": (112.0, 110.5,  99.0),
    "LAC": (111.5, 111.0,  98.5),
    "DAL": (112.5, 112.0,  97.0),
    "BKN": (112.0, 113.0, 100.0),
    "PHI": (111.0, 110.5,  97.5),
    "CLE": (111.5, 109.5,  96.5),
    "NYK": (110.0, 110.0,  96.0),
    "MEM": (112.0, 109.5, 101.0),
    "MIN": (113.0, 111.0,  99.5),
    "OKC": (112.5, 108.5,  99.0),
    # Lower seeds
    "TOR": (109.5, 111.5,  98.0),
    "CHI": (109.0, 112.0,  97.5),
    "ATL": (111.0, 113.0, 101.5),
    "NOP": (110.5, 112.5, 100.0),
    "SAC": (113.0, 114.0, 101.0),
    "IND": (111.5, 113.5, 100.5),
}
TEAMS = list(TEAM_PROFILES.keys())

SEASONS = [f"201{y}-1{y+1}" for y in range(4, 9)] + \
          [f"201{y}-2{y-9}" for y in range(9, 10)] + \
          ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]


def simulate_team_season_stats(team: str, season: str) -> dict:
    """Return a team's season-averaged stats with year-to-year noise."""
    off_mu, def_mu, pace_mu = TEAM_PROFILES[team]
    return {
        "team": team,
        "season": season,
        "season_off_rtg": rng.normal(off_mu, 1.5),
        "season_def_rtg": rng.normal(def_mu, 1.5),
        "season_pace": rng.normal(pace_mu, 1.2),
    }


def build_season_stats() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        for team in TEAMS:
            rows.append(simulate_team_season_stats(team, season))
    return pd.DataFrame(rows).set_index(["team", "season"])


def compute_win_prob(home: dict, away: dict) -> float:
    """
    Logistic function over net rating differential.
    Calibrated so ~5-point net rtg gap → ~75% win prob.
    Home court adds ~3 points (league historical average).
    """
    home_net = home["season_off_rtg"] - home["season_def_rtg"]
    away_net = away["season_off_rtg"] - away["season_def_rtg"]
    home_advantage = 3.0
    diff = (home_net - away_net) + home_advantage
    # logistic: k=0.25 → gentle curve
    prob = 1 / (1 + np.exp(-0.25 * diff))
    return float(prob)


def generate_games(season_stats: pd.DataFrame, n_games: int = 1200) -> pd.DataFrame:
    records = []
    teams_list = TEAMS

    for _ in range(n_games):
        season = rng.choice(SEASONS)
        home_team, away_team = rng.choice(teams_list, size=2, replace=False)

        try:
            h = season_stats.loc[(home_team, season)]
            a = season_stats.loc[(away_team, season)]
            # If MultiIndex returns a DataFrame (duplicate keys), take first row
            if isinstance(h, pd.DataFrame):
                h = h.iloc[0]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
        except KeyError:
            continue

        # Game-level stat noise (represents in-game variance)
        noise = lambda: rng.normal(0, 2.0)

        home_off_rtg = float(h["season_off_rtg"]) + noise()
        home_def_rtg = float(h["season_def_rtg"]) + noise()
        away_off_rtg = float(a["season_off_rtg"]) + noise()
        away_def_rtg = float(a["season_def_rtg"]) + noise()
        home_pace     = float(h["season_pace"]) + rng.normal(0, 1.0)
        away_pace     = float(a["season_pace"]) + rng.normal(0, 1.0)

        home_rest = rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05])
        away_rest = rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05])

        # Recent form: win% in last 10 games (correlated with net rtg)
        home_net = float(h["season_off_rtg"]) - float(h["season_def_rtg"])
        away_net = float(a["season_off_rtg"]) - float(a["season_def_rtg"])
        home_form = np.clip(rng.normal(0.5 + 0.03 * home_net, 0.15), 0, 1)
        away_form = np.clip(rng.normal(0.5 + 0.03 * away_net, 0.15), 0, 1)

        win_prob = compute_win_prob(
            {"season_off_rtg": home_off_rtg, "season_def_rtg": home_def_rtg},
            {"season_off_rtg": away_off_rtg, "season_def_rtg": away_def_rtg},
        )
        # Adjust for rest & form
        rest_edge = 0.01 * (home_rest - away_rest)
        form_edge = 0.05 * (home_form - away_form)
        win_prob = np.clip(win_prob + rest_edge + form_edge, 0.05, 0.95)

        home_wins = int(rng.random() < win_prob)

        records.append({
            "season": season,
            "home_team": home_team,
            "away_team": away_team,
            "home_off_rtg": round(home_off_rtg, 2),
            "home_def_rtg": round(home_def_rtg, 2),
            "away_off_rtg": round(away_off_rtg, 2),
            "away_def_rtg": round(away_def_rtg, 2),
            "home_pace": round(home_pace, 2),
            "away_pace": round(away_pace, 2),
            "home_rest_days": int(home_rest),
            "away_rest_days": int(away_rest),
            "home_form_l10": round(float(home_form), 3),
            "away_form_l10": round(float(away_form), 3),
            "home_wins": home_wins,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    season_stats = build_season_stats()
    games = generate_games(season_stats, n_games=1500)
    games.to_csv("data/playoff_games.csv", index=False)
    print(f"Generated {len(games)} games → data/playoff_games.csv")
    print(games.head())
    print(f"\nHome win rate: {games['home_wins'].mean():.3f}")
