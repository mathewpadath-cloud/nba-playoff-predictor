import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/playoff_games_real.csv')

df['net_rtg_diff'] = (df['home_off_rtg'] - df['home_def_rtg']) - (df['away_off_rtg'] - df['away_def_rtg'])
df['off_rtg_diff'] = df['home_off_rtg'] - df['away_off_rtg']
df['def_rtg_diff'] = df['home_def_rtg'] - df['away_def_rtg']
df['pace_diff']    = df['home_pace']    - df['away_pace']

FEATURES = ['net_rtg_diff', 'off_rtg_diff', 'def_rtg_diff', 'pace_diff']
X = df[FEATURES]
y = df['home_wins']

model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])
model.fit(X, y)

joblib.dump(model, 'model/playoff_predictor_real.pkl')

print(f'Accuracy: {accuracy_score(y, model.predict(X)):.3f}')
print(f'Games trained on: {len(df)}')
print('Model saved!')