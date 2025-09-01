import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pickle
from datetime import datetime
BASE_DIR = os.path.dirname(__file__)

# --- Load Models ---
models = {
    "D1": joblib.load(os.path.join(BASE_DIR, "../models/RandomForest_NoDraw_D1.joblib")),
    "E0": joblib.load(os.path.join(BASE_DIR, "../models/RandomForest_NoDraw_E0.joblib")),
    "F1": joblib.load(os.path.join(BASE_DIR, "../models/RandomForest_NoDraw_F1.joblib")),
    "I1": joblib.load(os.path.join(BASE_DIR, "../models/RandomForest_NoDraw_I1.joblib")),
    "SP1": joblib.load(os.path.join(BASE_DIR, "../models/RandomForest_NoDraw_SP1.joblib"))
}

# --- Load X_not_draw per league ---
X_not_draw_data = {
    "D1": pd.read_csv(os.path.join(BASE_DIR, "../data/train/X_not_draw_D1.csv")),
    "E0": pd.read_csv(os.path.join(BASE_DIR, "../data/train/X_not_draw_E0.csv")),
    "F1": pd.read_csv(os.path.join(BASE_DIR, "../data/train/X_not_draw_F1.csv")),
    "I1": pd.read_csv(os.path.join(BASE_DIR, "../data/train/X_not_draw_I1.csv")),
    "SP1": pd.read_csv(os.path.join(BASE_DIR, "../data/train/X_not_draw_SP1.csv"))
}

def prob(Date, HomeTeam, AwayTeam, league_code):
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

    X_not_draw = X_not_draw_data[league_code]
    final_model = models[league_code]
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
    ]].shift(1).rolling(5, min_periods=1).mean().iloc[-1]  
    
    
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
    ]].shift(1).rolling(5, min_periods=1).mean().iloc[-1]
    
    num_cols = X_not_draw.select_dtypes(include=np.number).columns
    home_profile.fillna(X_not_draw[num_cols].mean(), inplace=True)
    away_profile.fillna(X_not_draw[num_cols].mean(), inplace=True)

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
    cat_cols = ['Division', 'HomeTeam', 'AwayTeam']

    # Fixes types
    for col in match_features.columns:
        if col in cat_cols:
            match_features[col] = match_features[col].astype(str).fillna("Unknown")
        else:
            match_features[col] = pd.to_numeric(match_features[col], errors='coerce').fillna(0)

   

    y_proba = final_model.predict_proba(match_features.drop(columns=['MatchDate']))[0]
    p_home = y_proba[list(final_model.classes_).index(2)]
    p_away = y_proba[list(final_model.classes_).index(0)]
    delta = 0.03
    total = 1
    if abs(p_home - p_away) < delta:
        p_draw = max(delta - abs(p_home - p_away), 0)
    else:
        p_draw = 0

    total = p_home + p_away + p_draw
    p_home = round((p_home / total) * 100, 2)
    p_draw = round((p_draw / total) * 100, 2)
    p_away = round((p_away / total) * 100, 2)

    b = f'{home_team}: {p_home}% - Draw: {p_draw}% - {away_team}: {p_away}% '

    return b

# --- Streamlit ---
st.title("⚽ Match Predictor!")
st.markdown("""
Select the Home and Away teams, enter the date of the match, and click **Predict**  
to see the probability of Home win, Draw, and Away win.
""", unsafe_allow_html=True)

leagues = [("LaLiga", "SP1"), ("Premier League", "E0"), ("Serie A", "I1"), ("Bundesliga", "D1"), ("Ligue 1", "F1")]

league_name = st.selectbox("Select the league:", [l[0] for l in leagues])
league_code = next(code for name, code in leagues if name == league_name)

df_league = X_not_draw_data[league_code]
available_teams = sorted(pd.unique(df_league[['HomeTeam', 'AwayTeam']].values.ravel()))

with st.expander("See available teams"):
    st.dataframe(pd.DataFrame(available_teams, columns=["Teams"]))

home_team = st.selectbox("Select the Home Team:", available_teams, index=0)
away_options = [team for team in available_teams if team != home_team]
away_team = st.selectbox("Select the Away Team:", away_options, index=0)

date_str = st.text_input("Enter the date of the match (YYYY-MM-DD):", "")

if st.button("Predict"):
    if not home_team or not away_team or not date_str:
        st.warning("Please fill all fields!")
    else:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            prediction = prob(date_str, home_team, away_team, league_code)
            st.markdown(f"<h3 style='color:green'>Prediction:</h3><p style='font-size:20px'>{prediction}</p>", unsafe_allow_html=True)
        except ValueError:
            st.warning("Date must be in YYYY-MM-DD format.")