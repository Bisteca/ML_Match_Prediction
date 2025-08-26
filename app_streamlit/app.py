import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pickle

BASE_DIR = os.path.dirname(__file__)  # Diretório onde app.py está
model_path = os.path.join(BASE_DIR, "../models/final_model.pkl")

final_model = joblib.load(model_path)
X_not_draw_path = os.path.join(BASE_DIR, "../data/train/X_not_draw.csv")
X_not_draw = pd.read_csv(X_not_draw_path)


def prob(Date, HomeTeam, AwayTeam, df_league):
    """Predict probabilities for a match including optional draw adjustment.

    Args:
        Date (str): Match date in 'YYYY-MM-DD' format.
        HomeTeam (str): Name of the home team.
        AwayTeam (str): Name of the away team.
        delta (float): Threshold for adding draw probability. Default is 0.03.

    Returns:
        str: Formatted probabilities for home, draw, and away.
    """
    # --- Basic match info ---

    X_not_draw = df_league
    X_not_draw.drop(columns='Division')
    date = pd.to_datetime(Date)
    home_team = HomeTeam.title()
    away_team = AwayTeam.title()
    year, month, day = date.year, date.month, date.day
    dayofweek = date.dayofweek   
    is_weekend = int(dayofweek in [5, 6])


     # --- Create feature dataframe for the match ---
    prob_df = pd.DataFrame({
        'MatchDate': date,
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'Year': year,
        'IsWeekend': is_weekend, 
        'DayOfWeek_sin': np.sin(2 * np.pi * dayofweek / 7),
        'DayOfWeek_cos': np.cos(2 * np.pi * dayofweek / 7),
        'Month_sin': np.sin(2 * np.pi * month / 12),
        'Month_cos': np.cos(2 * np.pi * month / 12),
        'Day_sin': np.sin(2 * np.pi * day / 31),
        'Day_cos': np.cos(2 * np.pi * day / 31),
        'Season': year

    }, index=[0])

    
    # Home Team
    home_games = X_not_draw[X_not_draw['HomeTeam'] == home_team].sort_values('MatchDate')
    home_profile = home_games.loc[:, [
        'C_LTH', 'C_HTB', 'C_PHB', 'HomeElo', 'Form3Home','Form5Home','GF3Home',
        'GA3Home', 'GF5Home', 'GA5Home', 'WinStreakHome', 'DefeatStreakHome',
        'H2HHomeWins', 'GF_EMA3_Home', 'GF3HomeSTD', 'PointsAcumHome',
        'GF_Total_Home', 'GA_Total_Home', 'GD_total_Home', 'PointMeanHome',
        'ScoredGoalsMeanHome', 'ConcededGoalsMeanHome', 'GoalsDifferenceMeanHome',
        'WinHomeAcum', 'LossHomeAcum','WinRateHome', 'LossRateHome',
        'OddHome', 'ImpliedProbHome', 'HandiHome', 'MaxHome', 'BookieBiasHome'
    ]].shift(1).rolling(5).mean().iloc[-1]  # pega só o último perfil
    
    # Away Team 
    away_games = X_not_draw[X_not_draw['AwayTeam'] == away_team].sort_values('MatchDate')
    away_profile = away_games.loc[:, [
        'C_LTA', 'C_VHD', 'C_VAD', 'AwayElo', 'Form3Away','Form5Away','GF3Away',
        'GA3Away', 'GF5Away', 'GA5Away', 'WinStreakAway', 'DefeatStreakAway',
        'H2HAwayWins', 'GF_EMA3_Away','GF3AwaySTD', 'PointsAcumAway',
        'GF_Total_Away', 'GA_Total_Away', 'GD_total_Away', 'PointMeanAway',
        'ScoredGoalsMeanAway','ConcededGoalsMeanAway', 'GoalsDifferenceMeanAway',
        'WinAwayAcum', 'LossAwayAcum','WinRateAway','LossRateAway',
        'OddAway', 'ImpliedProbAway', 'HandiAway', 'MaxAway', 'BookieBiasAway'
    ]].shift(1).rolling(5).mean().iloc[-1]

    # --- Compute interaction features ---
    interactions = {
        'EloDifference': home_profile['HomeElo'] - away_profile['AwayElo'],
        'Form3Difference': home_profile['Form3Home'] - away_profile['Form3Away'],
        'Form5Difference': home_profile['Form5Home'] - away_profile['Form5Away'],
        'EloRatio': home_profile['HomeElo'] / (away_profile['AwayElo'] + 1e-6),
        'FormRatio': home_profile['Form3Home'] / (away_profile['Form3Away'] + 1),
        'GoalRateRatio': home_profile['ScoredGoalsMeanHome'] / (away_profile['ScoredGoalsMeanAway'] + 1),
        'WinRateDiff': home_profile['WinRateHome'] - away_profile['WinRateAway'],
        'PointsDiff': home_profile['PointsAcumHome'] - away_profile['PointsAcumAway'],
        'FormDiff': home_profile['Form5Home'] - away_profile['Form5Away'],
        'StreakDiff': home_profile['WinStreakHome'] - away_profile['DefeatStreakAway'],
        'ImpliedProbTotal': home_profile['ImpliedProbHome'] + away_profile['ImpliedProbAway'],
        'BookmakerMargin': (home_profile['ImpliedProbHome'] + away_profile['ImpliedProbAway']) - 1,
        'OddsDifference': home_profile["ImpliedProbHome"] - away_profile["ImpliedProbAway"],
        'Elo_ProbDiff': (home_profile["ImpliedProbHome"] - away_profile["ImpliedProbAway"]) * (home_profile['HomeElo'] - away_profile['AwayElo']),
        'OddSkew': (home_profile['OddHome'] - away_profile['OddAway']) / (home_profile['OddHome'] + away_profile['OddAway']),
        'FormVolatility': (home_profile['Form5Home'] - home_profile['Form3Home']) - (away_profile['Form5Away'] - away_profile['Form3Away']),
        'EloOddsGap': (home_profile['ImpliedProbHome'] - away_profile['ImpliedProbAway']) - ((home_profile['HomeElo'] / (away_profile['AwayElo'] + 1e-6)) / (1 + (home_profile['HomeElo'] / (away_profile['AwayElo'] + 1e-6)))),
        'Season': np.where(month >= 8, year, year - 1)
    }
    

    for i in ['HandiSize', 'Over25', 'Under25', 'MaxOver25', 'MaxUnder25']:
        home_mean = home_games[i].shift(1).mean()
        away_mean = away_games[i].shift(1).mean()
        interactions[i] =  (home_mean  + away_mean) / 2
 

    # Putting all together
    # --- Combine all features ---
    match_features = pd.DataFrame([{**prob_df.iloc[0].to_dict(), **home_profile.to_dict(), **away_profile.to_dict(), **interactions}])
    match_features = match_features.reindex(columns=X_not_draw.columns, fill_value=0)

    
    
    y_proba = final_model.predict_proba(match_features.drop(columns=['MatchDate']))[0]
    p_home, p_away = y_proba[1], y_proba[0]
    delta = 0.03
    total = 1
    if abs(p_home - p_away) < delta:
            p_draw = (abs(p_home - p_away)) / 2
            p_home -= p_draw / 2
            p_away -= p_draw / 2
    else:
            p_draw = 0 

    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = round((p_home/total) * 100, 2), round((p_draw/total) * 100, 2), round((p_away/total) * 100, 2)

    b = f'{home_team}: {p_home}% - Draw: {p_draw}% - {away_team}: {p_away}% '

    return b



st.title("Match Predictor!")
st.subheader("This is a match predictors, please put two teams and select the date that they are playing.")

league = st.selectbox(
    "Select the league:",
    ["LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"]
)


home_team = st.text_input("Enter the Home Team", key=f"home_{league}")
away_team = st.text_input("Enter the Away Team", key=f"away_{league}")
date = st.text_input("Enter the date of the game (YYYY/MM/DD)", key=f"date_{league}")

if league == "LaLiga":
    df_league = X_not_draw[X_not_draw['Division'].isin(['SP1'])]
elif league == "Premier League":
    df_league = X_not_draw[X_not_draw['Division'].isin(['E0'])]
elif league == "Serie A":
    df_league = X_not_draw[X_not_draw['Division'].isin(['I1'])]
elif league == "Bundesliga":
    df_league = X_not_draw[X_not_draw['Division'].isin(['D1'])]
elif league == "Ligue 1":
    df_league = X_not_draw[X_not_draw['Division'].isin(['F1'])]

available_teams = pd.unique(df_league[['HomeTeam', 'AwayTeam']].values.ravel())

with st.expander("See available teams"):
    st.write(available_teams)

if st.button("Predict", key=f"btn_{league}"):
    if not home_team or not away_team or not date:
        st.warning("Please fill all fields!")
    else:
        prediction = prob(date, home_team, away_team, df_league)
        st.markdown(
            f"<h3 style='color:green'>Prediction:</h3><p style='font-size:20px'>{prediction}</p>",
            unsafe_allow_html=True
        )




