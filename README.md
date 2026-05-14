# 🏀 NBA Playoff Game Predictor

A logistic regression model that predicts which team wins a playoff game based on team performance stats and game context. Given two teams, it outputs a win probability for each.

## Features Used
- **Net Rating** (offensive rating − defensive rating)
- **Offensive Rating** (points per 100 possessions)
- **Defensive Rating** (points allowed per 100 possessions)
- **Pace** (possessions per game)
- **Rest days** before game
- **Recent form** (win % in last 10 games)

All features are computed as **home minus away differentials** so the model learns relative advantages rather than raw numbers.

## Project Structure
```
nba_predictor/
├── data/
│   └── playoff_games.csv       # 1,500 generated playoff games
├── model/
│   ├── playoff_predictor.pkl   # trained logistic regression
│   └── team_averages.csv       # per-team stat averages for CLI
├── plots/                      # all generated charts
├── generate_data.py            # data generation script
├── notebook.ipynb              # full ML pipeline walkthrough
├── predict.py                  # CLI predictor
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nba-playoff-predictor.git
cd nba-playoff-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data + train model (run the notebook, or:)
python generate_data.py
```

## Usage

```bash
# Interactive mode
python predict.py

# Quick prediction (home team first)
python predict.py BOS MIA

# With rest day context
python predict.py DEN PHX --home-rest 3 --away-rest 1

# List all available teams
python predict.py --list-teams
```

## Model Results
- **Accuracy**: ~67%
- **ROC-AUC**: ~0.72
- Trained on 8 seasons, tested on 2 most recent (no data leakage)

## Swapping in Live Data
The dataset is currently synthetic but calibrated to real NBA distributions. To use real data via `nba_api`, replace the CSV loading in `notebook.ipynb` Step 1 with:

```python
from nba_api.stats.endpoints import leaguegamelog

log = leaguegamelog.LeagueGameLog(
    season='2022-23',
    season_type_all_star='Playoffs',
    player_or_team_abbreviation='T'
)
df = log.get_data_frames()[0]
```

Note: `nba_api` is rate-limited — add `time.sleep(1)` between requests when looping over seasons.
