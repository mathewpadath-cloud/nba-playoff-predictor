#!/usr/bin/env python3
"""
predict.py — NBA Playoff Game Predictor CLI
============================================
Usage
-----
  python predict.py                          # interactive wizard
  python predict.py BOS MIA                 # quick prediction (home team first)
  python predict.py BOS MIA --home-rest 3   # with rest-day flags
  python predict.py --list-teams            # show all available teams

Examples
--------
  python predict.py GSW LAL
  python predict.py DEN PHX --home-rest 3 --away-rest 1
"""

import argparse
import sys
import os
import joblib
import pandas as pd
import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, 'model', 'playoff_predictor.pkl')
TEAMS_PATH   = os.path.join(SCRIPT_DIR, 'model', 'team_averages.csv')

# ── Colour helpers (ANSI, gracefully disabled on Windows) ──────────────────
USE_COLOUR = sys.stdout.isatty() and os.name != 'nt'

def bold(s):   return f'\033[1m{s}\033[0m'  if USE_COLOUR else s
def green(s):  return f'\033[92m{s}\033[0m' if USE_COLOUR else s
def blue(s):   return f'\033[94m{s}\033[0m' if USE_COLOUR else s
def grey(s):   return f'\033[90m{s}\033[0m' if USE_COLOUR else s
def yellow(s): return f'\033[93m{s}\033[0m' if USE_COLOUR else s


# ── Model loading ──────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("❌  Model not found. Run the notebook first to generate model/playoff_predictor.pkl")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def load_team_stats():
    if not os.path.exists(TEAMS_PATH):
        print("❌  Team stats not found. Run the notebook first to generate model/team_averages.csv")
        sys.exit(1)
    df = pd.read_csv(TEAMS_PATH, index_col=0)
    return df


# ── Prediction logic ───────────────────────────────────────────────────────
FEATURES = ['net_rtg_diff', 'off_rtg_diff', 'def_rtg_diff',
            'pace_diff', 'rest_diff', 'form_diff']

def predict(home_team: str, away_team: str,
            team_stats: pd.DataFrame, model,
            home_rest: int = 2, away_rest: int = 2) -> dict:
    """
    Returns a dict with win probabilities and contributing factors.
    """
    lookup = {t.upper(): t for t in team_stats.index}

    hk = home_team.upper()
    ak = away_team.upper()

    if hk not in lookup:
        print(f"❌  Unknown team: '{home_team}'. Use --list-teams to see valid abbreviations.")
        sys.exit(1)
    if ak not in lookup:
        print(f"❌  Unknown team: '{away_team}'. Use --list-teams to see valid abbreviations.")
        sys.exit(1)

    h = team_stats.loc[lookup[hk]]
    a = team_stats.loc[lookup[ak]]

    h_net = h['off_rtg'] - h['def_rtg']
    a_net = a['off_rtg'] - a['def_rtg']

    row = {
        'net_rtg_diff': h_net - a_net,
        'off_rtg_diff': h['off_rtg'] - a['off_rtg'],
        'def_rtg_diff': h['def_rtg'] - a['def_rtg'],
        'pace_diff'   : h['pace']    - a['pace'],
        'rest_diff'   : home_rest    - away_rest,
        'form_diff'   : h['form_l10']- a['form_l10'],
    }
    X = pd.DataFrame([row])[FEATURES]
    prob_home = float(model.predict_proba(X)[0, 1])

    return {
        'home_team'  : lookup[hk],
        'away_team'  : lookup[ak],
        'home_prob'  : prob_home,
        'away_prob'  : 1 - prob_home,
        'home_rest'  : home_rest,
        'away_rest'  : away_rest,
        'home_net_rtg': round(h_net, 2),
        'away_net_rtg': round(a_net, 2),
        'features'   : row,
    }


# ── Display ────────────────────────────────────────────────────────────────
def probability_bar(prob: float, width: int = 40) -> str:
    filled = round(prob * width)
    bar    = '█' * filled + '░' * (width - filled)
    return bar


def print_result(result: dict):
    home  = result['home_team']
    away  = result['away_team']
    hp    = result['home_prob']
    ap    = result['away_prob']

    winner      = home if hp >= ap else away
    winner_prob = max(hp, ap)

    print()
    print(bold('━' * 54))
    print(bold(f'  🏀  {home:^10}  (home)  vs  {away:^10}  (away)'))
    print(bold('━' * 54))
    print()

    # Win probability bars
    hp_str = f'{hp*100:5.1f}%'
    ap_str = f'{ap*100:5.1f}%'

    print(f'  {home:<6}  {blue(probability_bar(hp))}  {bold(hp_str)}')
    print(f'  {away:<6}  {blue(probability_bar(ap))}  {bold(ap_str)}')
    print()

    # Verdict
    confidence = 'decisive' if winner_prob > 0.70 else \
                 'moderate' if winner_prob > 0.58 else 'slight'
    verdict = green(f'  Predicted winner: {bold(winner)}  ({winner_prob*100:.1f}%  –  {confidence} edge)')
    print(verdict)
    print()

    # Context
    print(grey('  ── Context ──────────────────────────────────────'))
    print(grey(f'  Net rating  :  {home} {result["home_net_rtg"]:+.2f}  |  '
               f'{away} {result["away_net_rtg"]:+.2f}'))
    print(grey(f'  Rest days   :  {home} {result["home_rest"]}d  |  {away} {result["away_rest"]}d'))

    # Key driver
    f = result['features']
    drivers = [
        ('net_rtg_diff', 'Net rating differential'),
        ('off_rtg_diff', 'Offensive rating edge'),
        ('def_rtg_diff', 'Defensive rating edge'),
        ('rest_diff',    'Rest advantage'),
        ('form_diff',    'Recent form advantage'),
    ]
    top_driver = max(drivers, key=lambda x: abs(f[x[0]]))[1]
    print(grey(f'  Top factor  :  {top_driver}'))
    print(grey('  ────────────────────────────────────────────────'))
    print()


# ── Interactive wizard ─────────────────────────────────────────────────────
def interactive_mode(team_stats: pd.DataFrame, model):
    teams = sorted(team_stats.index.tolist())
    print()
    print(bold('🏀  NBA Playoff Game Predictor'))
    print(grey('   (type a team abbreviation, e.g. BOS, GSW, LAL)'))
    print()
    print(grey('Available teams: ') + ', '.join(teams))
    print()

    home = input(yellow('  Home team abbreviation: ')).strip().upper()
    away = input(yellow('  Away team abbreviation: ')).strip().upper()

    try:
        home_rest = int(input(yellow('  Home team rest days [default 2]: ') or '2').strip() or '2')
        away_rest = int(input(yellow('  Away team rest days [default 2]: ') or '2').strip() or '2')
    except ValueError:
        home_rest, away_rest = 2, 2

    result = predict(home, away, team_stats, model, home_rest, away_rest)
    print_result(result)

def get_live_stats(team_abbrev):
    """Pull current 2025-26 playoff stats for a team from NBA API."""
    from nba_api.stats.endpoints import teamdashboardbygeneralsplits
    from nba_api.stats.static import teams as nba_teams
    import time

    all_teams = nba_teams.get_teams()
    lookup = {t['abbreviation'].upper(): t['id'] for t in all_teams}
    
    abbrev = team_abbrev.upper()
    if abbrev not in lookup:
        print(f"❌  Unknown team: {team_abbrev}")
        sys.exit(1)
    
    time.sleep(1)
    df = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=lookup[abbrev],
        season='2025-26',
        season_type_all_star='Playoffs',
        measure_type_detailed_defense='Advanced',
        timeout=60
    ).get_data_frames()[0]

    if df.empty:
        print(f"❌  No 2025-26 playoff data found for {team_abbrev}")
        sys.exit(1)

    return {
        'off_rtg': float(df['OFF_RATING'].iloc[0]),
        'def_rtg': float(df['DEF_RATING'].iloc[0]),
        'pace':    float(df['PACE'].iloc[0]),
        'form_l10': 0.0,
    }

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='NBA Playoff Game Predictor — logistic regression model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('home', nargs='?', help='Home team abbreviation (e.g. BOS)')
    parser.add_argument('away', nargs='?', help='Away team abbreviation (e.g. MIA)')
    parser.add_argument('--home-rest', type=int, default=2, metavar='N',
                        help='Home team rest days before game (default: 2)')
    parser.add_argument('--away-rest', type=int, default=2, metavar='N',
                        help='Away team rest days before game (default: 2)')
    parser.add_argument('--live', action='store_true',
                        help='Pull live 2025-26 stats from NBA API instead of historical averages')
    parser.add_argument('--list-teams', action='store_true',
                        help='List all available team abbreviations and exit')

    args = parser.parse_args()

    model      = load_model()
    team_stats = load_team_stats()

    if args.list_teams:
        print('\nAvailable teams:')
        teams = sorted(team_stats.index.tolist())
        for i, t in enumerate(teams):
            end = '\n' if (i + 1) % 6 == 0 else '  '
            print(f'  {t:<6}', end=end)
        print('\n')
        sys.exit(0)

    if args.home and args.away:
        if args.live:
            print("Fetching live 2025-26 stats...")
            home_stats_live = get_live_stats(args.home)
            away_stats_live = get_live_stats(args.away)
            live_df = pd.DataFrame([{**{f'home_{k}': v for k, v in home_stats_live.items()},
                                      **{f'away_{k}': v for k, v in away_stats_live.items()}}])
            live_team_stats = pd.DataFrame({
                args.home.upper(): home_stats_live,
                args.away.upper(): away_stats_live,
            }).T
            live_team_stats.index.name = None
            result = predict(args.home, args.away, live_team_stats, model,
                             args.home_rest, args.away_rest)
        else:
            result = predict(args.home, args.away, team_stats, model,
                             args.home_rest, args.away_rest)
        print_result(result)
    else:
        interactive_mode(team_stats, model)


if __name__ == '__main__':
    main()