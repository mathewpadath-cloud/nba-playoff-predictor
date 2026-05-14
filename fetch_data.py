"""
fetch_data.py
-------------
Pulls real NBA playoff data from nba_api for seasons 2014-15 to 2023-24.
Saves to data/playoff_games_real.csv
"""

import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, teamdashboardbygeneralsplits
from nba_api.stats.static import teams

SEASONS = [
    '2014-15', '2015-16', '2016-17', '2017-18', '2018-19',
    '2019-20', '2020-21', '2021-22', '2022-23', '2023-24'
]

def get_all_teams():
    """Returns a dict of abbreviation -> team_id for all NBA teams."""
    all_teams = teams.get_teams()
    return {t['abbreviation']: t['id'] for t in all_teams}

def get_team_advanced_stats(team_id, season):
    """Pull off rating, def rating, pace for one team in one season."""
    for attempt in range(3):  # retry up to 3 times
        try:
            time.sleep(2)  # increased to 2 seconds
            df = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                team_id=team_id,
                season=season,
                season_type_all_star='Playoffs',
                measure_type_detailed_defense='Advanced',
                timeout=60  # increased timeout
            ).get_data_frames()[0]
            
            if df.empty:
                return None
            
            return {
                'off_rtg': df['OFF_RATING'].iloc[0],
                'def_rtg': df['DEF_RATING'].iloc[0],
                'pace':    df['PACE'].iloc[0],
            }
        except Exception as e:
            print(f"  Attempt {attempt+1} failed, retrying...")
            time.sleep(5)  # wait longer before retry
    return None
    
def get_game_log(season):
    """Pull all playoff games for a season with winner info."""
    time.sleep(1)
    df = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star='Playoffs',
        player_or_team_abbreviation='T',
        timeout=30
    ).get_data_frames()[0]
    return df

def build_dataset(seasons):
    """Loop through seasons, pull all data, combine into one dataframe."""
    all_games = []
    team_ids = get_all_teams()
    
    for season in seasons:
        print(f"Fetching {season}...")
        
        # Get game log for this season
        game_log = get_game_log(season)
        
        # Get advanced stats for every team
        team_stats = {}
        for abbrev, team_id in team_ids.items():
            stats = get_team_advanced_stats(team_id, season)
            if stats:
                team_stats[abbrev] = stats
        
        # Find home/away games by looking at MATCHUP column
        # Home games contain 'vs.', away games contain '@'
        home_games = game_log[game_log['MATCHUP'].str.contains('vs\.')]
        
        for _, home_row in home_games.iterrows():
            home_team = home_row['TEAM_ABBREVIATION']
            game_id = home_row['GAME_ID']
            
            # Find the matching away team row
            away_row = game_log[game_log['GAME_ID'] == game_id]
            away_row = away_row[away_row['TEAM_ABBREVIATION'] != home_team]
            
            if away_row.empty:
                continue
            away_row = away_row.iloc[0]
            away_team = away_row['TEAM_ABBREVIATION']
            
            # Skip if we don't have advanced stats for either team
            if home_team not in team_stats or away_team not in team_stats:
                continue
            
            h = team_stats[home_team]
            a = team_stats[away_team]
            
            all_games.append({
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'home_off_rtg': h['off_rtg'],
                'home_def_rtg': h['def_rtg'],
                'away_off_rtg': a['off_rtg'],
                'away_def_rtg': a['def_rtg'],
                'home_pace': h['pace'],
                'away_pace': a['pace'],
                'home_wins': 1 if home_row['WL'] == 'W' else 0,
            })
    
    return pd.DataFrame(all_games)

if __name__ == '__main__':
    print("Starting NBA playoff data fetch...")
    print("This will take 10-15 minutes due to API rate limiting. Don't close the terminal.")
    
    df = build_dataset(SEASONS)
    
    df.to_csv('data/playoff_games_real.csv', index=False)
    print(f"\nDone! Saved {len(df)} games to data/playoff_games_real.csv")
    print(df.head())