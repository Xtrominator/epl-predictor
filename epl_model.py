# EPL match predictor using XGBoost
# Train on 2017-2023 seasons, test + betting backtest on 2023-24
# Data comes from football-data.co.uk (free CSV files)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score

train_seasons = ["1718", "1819", "1920", "2021", "2122", "2223"]
test_season = "2324"


# ---------------------------------------------------------
# Step 1: load the data
# ---------------------------------------------------------
def load_season(code):
    url = "https://www.football-data.co.uk/mmz4281/" + code + "/E0.csv"
    df = pd.read_csv(url)
    # only keep the columns we need
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
             "B365H", "B365D", "B365A"]]
    df = df.dropna()  # some files have empty rows at the bottom
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["Season"] = code
    return df


# ---------------------------------------------------------
# Step 2: make the features
# ---------------------------------------------------------
def make_features(season_df):
    """Goes through one season match by match.
    Features only use matches BEFORE the current one (otherwise
    the model would be cheating by seeing the future)."""
    season_df = season_df.sort_values("Date").reset_index(drop=True)

    points = {}       # team -> total points so far
    goal_diff = {}    # team -> goal difference so far
    last5 = {}        # team -> list of points from last matches
    streak = {}       # team -> games in a row without losing

    rows = []

    for i in range(len(season_df)):
        match = season_df.iloc[i]
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        # make sure both teams are in our dictionaries
        for team in [home, away]:
            if team not in points:
                points[team] = 0
                goal_diff[team] = 0
                last5[team] = []
                streak[team] = 0

        # current league table (sort by points, then goal difference)
        table = sorted(points.keys(),
                       key=lambda t: (points[t], goal_diff[t]),
                       reverse=True)
        home_pos = table.index(home) + 1
        away_pos = table.index(away) + 1

        # form = points from last 5 games
        home_form = sum(last5[home][-5:])
        away_form = sum(last5[away][-5:])

        # betting odds -> implied probabilities
        # 1/odds gives a probability but they add up to more than 1
        # (that extra is the bookmaker's cut), so we divide by the total
        raw_h = 1 / match["B365H"]
        raw_d = 1 / match["B365D"]
        raw_a = 1 / match["B365A"]
        total = raw_h + raw_d + raw_a

        rows.append({
            "prob_h": raw_h / total,
            "prob_d": raw_d / total,
            "prob_a": raw_a / total,
            "home_form": home_form,
            "away_form": away_form,
            "home_pos": home_pos,
            "away_pos": away_pos,
            "pos_diff": away_pos - home_pos,
            "home_streak": streak[home],
            "away_streak": streak[away],
        })

        # now update everything with this match's result
        hg = match["FTHG"]
        ag = match["FTAG"]
        result = match["FTR"]

        goal_diff[home] += hg - ag
        goal_diff[away] += ag - hg

        if result == "H":
            home_pts, away_pts = 3, 0
        elif result == "A":
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1

        points[home] += home_pts
        points[away] += away_pts
        last5[home].append(home_pts)
        last5[away].append(away_pts)

        streak[home] = streak[home] + 1 if home_pts > 0 else 0
        streak[away] = streak[away] + 1 if away_pts > 0 else 0

    features = pd.DataFrame(rows)
    features["Date"] = season_df["Date"]
    features["Result"] = season_df["FTR"].map({"H": 0, "D": 1, "A": 2})
    features["odds_h"] = season_df["B365H"]
    features["odds_d"] = season_df["B365D"]
    features["odds_a"] = season_df["B365A"]
    return features


def build_dataset(seasons):
    all_seasons = []
    for s in seasons:
        print("loading", s)
        df = load_season(s)
        all_seasons.append(make_features(df))
    return pd.concat(all_seasons, ignore_index=True)


feature_cols = ["prob_h", "prob_d", "prob_a", "home_form", "away_form",
                "home_pos", "away_pos", "pos_diff", "home_streak", "away_streak"]

train = build_dataset(train_seasons)
test = build_dataset([test_season])

X_train = train[feature_cols]
y_train = train["Result"]
X_test = test[feature_cols]
y_test = test["Result"]
print("train matches:", len(train), " test matches:", len(test))


# ---------------------------------------------------------
# Step 3: train the model + tune hyperparameters
# ---------------------------------------------------------
param_grid = {
    "max_depth": [2, 3, 4],
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.03, 0.1],
}

# TimeSeriesSplit keeps older matches in training and newer in validation
search = GridSearchCV(
    XGBClassifier(eval_metric="mlogloss"),
    param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring="accuracy",
)
search.fit(X_train, y_train)
print("best params:", search.best_params_)

model = search.best_estimator_
predictions = model.predict(X_test)
model_acc = accuracy_score(y_test, predictions)


# ---------------------------------------------------------
# Step 4: compare to the "just pick the favorite" baseline
# ---------------------------------------------------------
# favorite = outcome with the lowest odds (highest implied probability)
favorite = X_test[["prob_h", "prob_d", "prob_a"]].values.argmax(axis=1)
baseline_acc = accuracy_score(y_test, favorite)

print("model accuracy:    ", round(model_acc, 3))
print("favorite baseline: ", round(baseline_acc, 3))


# ---------------------------------------------------------
# Step 5: betting backtest on 2023-24
# ---------------------------------------------------------
# idea: if model_prob * odds > 1 then the bet has positive "value"
# only bet when the value is at least 5% and stake 2% of bankroll
probs = model.predict_proba(X_test)
odds = test[["odds_h", "odds_d", "odds_a"]].values

start_bankroll = 1000
bankroll = start_bankroll
bankroll_history = [bankroll]
total_staked = 0
num_bets = 0
wins = 0

min_value = 0.05
stake_percent = 0.02

for i in range(len(test)):
    # value of each of the 3 outcomes
    values = probs[i] * odds[i] - 1
    best = values.argmax()

    if values[best] > min_value:
        stake = bankroll * stake_percent
        total_staked += stake
        num_bets += 1

        if test["Result"].iloc[i] == best:
            bankroll += stake * (odds[i][best] - 1)
            wins += 1
        else:
            bankroll -= stake

    bankroll_history.append(bankroll)

profit = bankroll - start_bankroll
roi = profit / total_staked * 100 if total_staked > 0 else 0

print("bets placed:", num_bets, " wins:", wins)
print("final bankroll:", round(bankroll, 2))
print("ROI:", round(roi, 2), "%")


# ---------------------------------------------------------
# Step 6: plot the bankroll
# ---------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(bankroll_history)
plt.axhline(start_bankroll, color="gray", linestyle="--", label="starting bankroll")
plt.title("Bankroll during 2023-24 season (value betting)")
plt.xlabel("Match number")
plt.ylabel("Bankroll")
plt.legend()
plt.savefig("bankroll.png")
plt.show()
