import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import date

def process_data():
    """
    Reads CSV files from data/raw, processes them (if needed), 
    and saves the final dataframe to data/processed.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_file = os.path.join(base_dir, '..', 'data', 'raw', 'Matches.csv')
    df = pd.read_csv(raw_file)



    # Feature engineering
    leagues = ['SP1', 'E0', 'I1', 'D1', 'F1']  # SP1=LaLiga, E0=Premier League, I1=Serie A, D1=Bundesliga, F1=Ligue 1
    df = df[df['Division'].isin(leagues)]
    df.drop(columns=['MatchTime'], inplace= True )
    
    le = LabelEncoder()
    df['Results'] = le.fit_transform(df['FTResult'])


    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df['Year'] = df['MatchDate'].dt.year 
    df['Month'] = df['MatchDate'].dt.month 
    df['Day'] = df['MatchDate'].dt.day 
    df['DayOfWeek'] = df['MatchDate'].dt.dayofweek 
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    df['EloDifference'] = df['HomeElo'] - df['AwayElo']

    df['ImpliedProbHome'] = 1 / df['OddHome'] 
    df['ImpliedProbDraw'] = 1 / df['OddDraw'] 
    df['ImpliedProbAway'] = 1 / df['OddAway'] 

    df['Home_odds'] = 1 / (df['ImpliedProbHome'] + df['ImpliedProbDraw']) 
    df['Away_odds'] = 1/ (df['ImpliedProbAway'] + df['ImpliedProbDraw']) 
    df['Home_Away_Odds'] = 1 / (df['ImpliedProbHome'] + df['ImpliedProbAway'])

    df['Form3Difference'] = df['Form3Home'] - df['Form3Away'] 
    df['Form5Difference'] = df['Form5Home'] - df['Form5Away'] 

    df['ImpliedProbTotal'] = df['ImpliedProbHome']  + df['ImpliedProbAway'] 

    df['BookmakerMargin'] = df['ImpliedProbTotal'] - 1 


    df2 = df.sort_values(['HomeTeam', 'MatchDate'])
    df3 = df.sort_values(['AwayTeam', 'MatchDate'])

    df['GF3Home'] = df2.groupby('HomeTeam')['FTHome'].shift(1).rolling(3).sum().reindex(df.index)
    df['GF3Away'] = df3.groupby('AwayTeam')['FTAway'].shift(1).rolling(3).sum().reindex(df.index)


    df['GA3Home'] = df2.groupby('HomeTeam')['FTAway'].shift(1).rolling(3).sum().reindex(df.index)
    df['GA3Away'] = df3.groupby('AwayTeam')['FTHome'].shift(1).rolling(3).sum().reindex(df.index)


    df['GF5Home'] =  df2.groupby('HomeTeam')['FTHome'].shift(1).rolling(5).sum().reindex(df.index)
    df['GF5Away'] = df3.groupby('AwayTeam')['FTAway'].shift(1).rolling(5).sum().reindex(df.index)

    df['GA5Home'] = df2.groupby('HomeTeam')['FTAway'].shift(1).rolling(5).sum().reindex(df.index)
    df['GA5Away'] = df3.groupby('AwayTeam')['FTHome'].shift(1).rolling(5).sum().reindex(df.index)

    df[['GF3Away', 'GF3Home', 'GF5Away', 'GF5Home','GA5Away', 'GA5Home', 'GA3Home', 'GA3Away']] = df[['GF3Away', 'GF3Home', 'GF5Away', 'GF5Home','GA5Away', 'GA5Home', 'GA3Home', 'GA3Away']].fillna(0)
    


    winshome = df2.groupby('HomeTeam')['Results'].transform(
    lambda x: x.eq(2).astype(int)
                    .groupby((x != 2).cumsum())  
                    .cumsum()
                    .shift(1)
    ).fillna(0)

    df['WinStreakHome'] = winshome.reindex(df.index)

    winsaway = df3.groupby('AwayTeam')['Results'].transform(
        lambda x: x.eq(0).astype(int)
                        .groupby((x != 0).cumsum())  
                        .cumsum()
                        .shift(1)
    ).fillna(0)
    df['WinStreakAway'] = winsaway.reindex(df.index)


    drawhome = df2.groupby('HomeTeam')['Results'].transform(
        lambda x: x.eq(1).astype(int)
                        .groupby((x != 1).cumsum())  
                        .cumsum()
                        .shift(1)
    ).fillna(0)
    df['DrawStreakHome'] = drawhome.reindex(df.index)

    drawaway = df3.groupby('AwayTeam')['Results'].transform(
        lambda x: x.eq(1).astype(int)
                        .groupby((x != 1).cumsum())  
                        .cumsum()
                        .shift(1)
    ).fillna(0)
    df['DrawStreakAway'] = drawaway.reindex(df.index)



    defeathome = df2.groupby('HomeTeam')['Results'].transform(
        lambda x: x.eq(0).astype(int)
                        .groupby((x != 0).cumsum())  
                        .cumsum()
                        .shift(1)
    ).fillna(0)
    df['DefeatStreakHome'] = defeathome.reindex(df.index)

    defeataway = df3.groupby('AwayTeam')['Results'].transform(
        lambda x: x.eq(2).astype(int)
                        .groupby((x != 2).cumsum())  
                        .cumsum()
                        .shift(1)
    ).fillna(0)
    df['DefeatStreakAway'] = defeataway.reindex(df.index)

    df4 = df.sort_values(['HomeTeam', 'AwayTeam', 'MatchDate'])
    h2hhome =  df4.groupby(['HomeTeam', 'AwayTeam'])['Results'].transform(lambda x: (x.shift(1) == 2).rolling(5, min_periods=1).sum())
    df['H2HHomeWins'] = h2hhome.reindex(df.index)

    h2haway =  df4.groupby(['AwayTeam','HomeTeam'])['Results'].transform(lambda x: (x.shift(1) == 0).rolling(5, min_periods=1).sum())
    df['H2HAwayWins'] = h2haway.reindex(df.index)


    emahome = df2.groupby('HomeTeam')['FTHome'].shift(1).ewm(span=3).mean()
    df['GF_EMA3_Home'] = emahome.reindex(df.index)

    emaaway = df3.groupby('AwayTeam')['FTAway'].shift(1).ewm(span=3).mean()
    df['GF_EMA3_Away'] = emaaway.reindex(df.index)


    goalhome = df2.groupby('HomeTeam')['FTHome'].shift(1).rolling(3).std()
    df['GF3HomeSTD'] = goalhome.reindex(df.index).fillna(0)
    goalaway = df3.groupby('AwayTeam')['FTAway'].shift(1).rolling(3).std()
    df['GF3AwaySTD'] = goalaway.reindex(df.index).fillna(0)

    points_home = {2:3, 1:1, 0:0}
    df['PointsHome'] = df['Results'].map(points_home)

    points_away = {0:3, 1:1, 2:0}
    df['PointsAway'] = df['Results'].map(points_away)

    df['Season'] = np.where(df['Month'] >= 8, df['Year'], df['Year'] - 1)

    acumhome = df.groupby(['HomeTeam', 'Season'])['PointsHome'].transform(lambda x: x.shift(1).cumsum())
    df['PointsAcumHome'] = acumhome.reindex(df.index).fillna(0)
    acumaway = df.groupby(['AwayTeam', 'Season'])['PointsAway'].transform(lambda x: x.shift(1).cumsum())
    df['PointsAcumAway'] = acumaway.reindex(df.index).fillna(0)

    df2 = df.sort_values(['HomeTeam', 'MatchDate'])
    df3 = df.sort_values(['AwayTeam', 'MatchDate'])
    gf_home = df2.groupby(['HomeTeam', 'Season'])['FTHome'].transform(lambda x: x.shift(1).cumsum())
    df['GF_Total_Home'] = gf_home.reindex(df.index).fillna(0)

    gf_away = df3.groupby(['AwayTeam', 'Season'])['FTAway'].transform(lambda x: x.shift(1).cumsum())
    df['GF_Total_Away'] = gf_away.reindex(df.index).fillna(0)

    gf_home = df2.groupby(['HomeTeam', 'Season'])['FTAway'].transform(lambda x: x.shift(1).cumsum())
    df['GA_Total_Home'] = gf_home.reindex(df.index).fillna(0)

    gf_away = df3.groupby(['AwayTeam', 'Season'])['FTHome'].transform(lambda x: x.shift(1).cumsum())
    df['GA_Total_Away'] = gf_away.reindex(df.index).fillna(0)

    df['GD_total_Home'] = df['GF_Total_Home'] - df['GA_Total_Home']
    df['GD_total_Away'] = df['GF_Total_Away'] - df['GA_Total_Away']


    df['GameCount'] = 1
    game_played_home = df.groupby(['HomeTeam', 'Season'])['GameCount'].transform(lambda x: x.shift(1).cumsum())
    game_played_away = df.groupby(['AwayTeam', 'Season'])['GameCount'].transform(lambda x: x.shift(1).cumsum())

    df['PointMeanHome'] = (df['PointsAcumHome'] / game_played_home).fillna(0)

    df['PointMeanAway'] = (df['PointsAcumAway'] / game_played_away).fillna(0)

    df['ScoredGoalsMeanHome']= (df['GF_Total_Home'] / game_played_home).fillna(0)

    df['ScoredGoalsMeanAway']= (df['GF_Total_Away'] / game_played_away).fillna(0)

    df['ConcededGoalsMeanHome']= (df['GA_Total_Home']/ game_played_home).fillna(0)

    df['ConcededGoalsMeanAway']= (df['GA_Total_Away'] / game_played_away).fillna(0)

    df['GoalsDifferenceMeanHome']= (df['GD_total_Home']/ game_played_home).fillna(0)

    df['GoalsDifferenceMeanAway']= (df['GD_total_Away'] / game_played_away).fillna(0)

    df['WinHome'] = (df['FTResult'] == 'H').astype(int)
    df['Draw'] = (df['FTResult'] == 'D').astype(int)
    df['WinAway'] = (df['FTResult'] == 'A').astype(int)

    df['WinHomeAcum'] = df.groupby(['HomeTeam', 'Season'])['WinHome'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    df['WinAwayAcum'] = df.groupby(['AwayTeam', 'Season'])['WinAway'].transform(lambda x: x.shift(1).cumsum()).fillna(0)

    df['DrawHomeAcum'] = df.groupby(['HomeTeam', 'Season'])['Draw'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    df['DrawAwayAcum'] = df.groupby(['AwayTeam', 'Season'])['Draw'].transform(lambda x: x.shift(1).cumsum()).fillna(0)

    df['LossHome'] = (df['FTResult'] == 'A').astype(int)
    df['LossAway'] = (df['FTResult'] == 'H').astype(int)

    df['LossHomeAcum'] = df.groupby(['HomeTeam', 'Season'])['LossHome'].transform(lambda x: x.shift(1).cumsum()).fillna(0)

    df['LossAwayAcum'] = df.groupby(['AwayTeam', 'Season'])['LossAway'].transform(lambda x: x.shift(1).cumsum()).fillna(0)

    df['WinRateHome'] = (df['WinHomeAcum']  / game_played_home).fillna(0)

    df['WinRateAway'] = (df['WinAwayAcum'] / game_played_away).fillna(0)

    df['DrawRateHome'] = (df['DrawHomeAcum'] / game_played_home).fillna(0)

    df['DrawRateAway'] = (df['DrawAwayAcum'] / game_played_away).fillna(0)

    df['LossRateHome'] = (df['LossHomeAcum'] / game_played_home).fillna(0)

    df['LossRateAway'] = (df['LossAwayAcum'] / game_played_away).fillna(0)


    df["OddsDifference"] = df["ImpliedProbHome"] - df["ImpliedProbAway"]

    df["EloRatio"] = df["HomeElo"] / (df["AwayElo"] + 1e-6)


    df["FormRatio"] = df["Form3Home"] / (df["Form3Away"] + 1)

    df["GoalRateRatio"] = df["ScoredGoalsMeanHome"] / (df["ScoredGoalsMeanAway"] + 1)

    df["WinRateDiff"] = df["WinRateHome"] - df["WinRateAway"]

    df["PointsDiff"] = df["PointsAcumHome"] - df["PointsAcumAway"]

    df["FormDiff"] = df["Form5Home"] - df["Form5Away"]


    df["Elo_ProbDiff"] = (df["ImpliedProbHome"] - df["ImpliedProbAway"]) * df["EloDifference"]
    df["FormOddsDiff"] = df["Form5Difference"] * (df["ImpliedProbHome"] - df["ImpliedProbAway"])
    df["StreakDiff"] = df["WinStreakHome"] - df["DefeatStreakAway"]
    df["BookieBiasHome"] = df["ImpliedProbHome"] - df["WinRateHome"]

    df["BookieBiasAway"] = df["ImpliedProbAway"] - df["WinRateAway"]
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 31)


    df['EloOddsGap'] = (df['ImpliedProbHome'] - df['ImpliedProbAway']) - (df['EloRatio'] / (1+df['EloRatio']))

    df['FormVolatility'] = (df['Form5Home'] - df['Form3Home']) - (df['Form5Away'] - df['Form3Away'])

    df['OddSkew'] = (df['OddHome'] - df['OddAway']) / (df['OddHome'] + df['OddAway'])


    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    processed_file = os.path.join(processed_dir, 'processed_data.csv')
    df.to_csv(processed_file, index=False)
    print(f'Processed data saved to {processed_file}')

if __name__ == "__main__":
    process_data()
