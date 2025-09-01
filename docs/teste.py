import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit


base_dir = os.path.dirname(os.path.abspath(__file__))
raw_file = os.path.join(base_dir, '..', 'data', 'processed', 'processed_data.csv')
df = pd.read_csv(raw_file)


df = df[df['MatchDate'] > '2010-08']


counts_home = df['HomeTeam'].value_counts()
counts_away = df['AwayTeam'].value_counts()

valid_teams = counts_home[counts_home >= 20].index.intersection(counts_away[counts_away >= 20].index)
df = df[(df['HomeTeam'].isin(valid_teams)) & (df['AwayTeam'].isin(valid_teams))]


X = df[['Division','MatchDate', 'HomeTeam', 'AwayTeam', 'Year',  'IsWeekend',
        'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB',
    'HomeElo', 'AwayElo', 'EloDifference', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'Form3Difference', 'Form5Difference',
    'GF3Home', 'GF3Away', 'GA3Home', 'GA3Away', 'GF5Home', 'GF5Away', 'GA5Home', 'GA5Away',
    'OddHome', 'OddDraw', 'OddAway', 'ImpliedProbHome', 'ImpliedProbDraw', 'ImpliedProbAway', 'BookmakerMargin',
    'HandiSize', 'HandiHome', 'HandiAway', 
    'MaxHome', 'MaxDraw', 'MaxAway', 'Over25', 'Under25', 'MaxOver25', 'MaxUnder25', 
    'WinStreakHome', 'WinStreakAway', 'DrawStreakHome', 'DrawStreakAway', 'DefeatStreakHome', 'DefeatStreakAway', 'H2HHomeWins', 'H2HAwayWins', 'GF_EMA3_Home','GF_EMA3_Away', 'GF3HomeSTD',
    'GF3AwaySTD', 'Season', 'PointsAcumHome', 'PointsAcumAway', 
    'GF_Total_Home', 'GF_Total_Away', 'GA_Total_Home', 'GA_Total_Away', 
    'GD_total_Home', 'GD_total_Away', 'PointMeanHome', 'PointMeanAway', 'ScoredGoalsMeanHome', 'ScoredGoalsMeanAway', 'ConcededGoalsMeanHome', 'ConcededGoalsMeanAway',
    'GoalsDifferenceMeanHome','GoalsDifferenceMeanAway','WinHomeAcum','WinAwayAcum','DrawHomeAcum','DrawAwayAcum','LossHomeAcum', 'LossAwayAcum', 'WinRateHome','WinRateAway','DrawRateHome',
    'DrawRateAway','LossRateHome','LossRateAway', 
    "OddsDifference", "EloRatio", "FormRatio", "GoalRateRatio", "WinRateDiff", "PointsDiff", "FormDiff", "Elo_ProbDiff", "FormOddsDiff", "StreakDiff", "BookieBiasHome", "DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos", "Day_sin", "Day_cos",
    'OddSkew', 'FormVolatility', 'EloOddsGap', 'ImpliedProbTotal', 'BookieBiasAway'
]]
y = df['Results']
    


cols_draw = ['DrawStreakHome', 'DrawStreakAway', 
            'DrawHomeAcum', 'DrawAwayAcum',
            'DrawRateHome', 'DrawRateAway',
            'OddDraw', 'ImpliedProbDraw', 'MaxDraw']
# Dropping Draws
not_draw= y != 1
X_not_draw = X[not_draw].copy()
X_not_draw = X_not_draw.drop(columns=cols_draw)
y_not_draw = y[not_draw].copy()


X_train_nd = X_not_draw[X_not_draw['MatchDate'] < '2021-07'].drop(columns='MatchDate')
X_test_nd = X_not_draw[X_not_draw['MatchDate'] > '2021-08'].drop(columns='MatchDate')
y_train_nd = y_not_draw.loc[X_train_nd.index]
y_test_nd = y_not_draw.loc[X_test_nd.index]





X_train_nd.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_nd.fillna(0, inplace=True)

X_test_nd.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_nd.fillna(0, inplace=True)

# Categorical colums
cat_cols = ['Division', 'HomeTeam', 'AwayTeam']



# Numerical Colums
num_cols= ['Year',  'IsWeekend',
        'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB',
        'HomeElo', 'AwayElo', 'EloDifference', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'Form3Difference', 'Form5Difference',
        'GF3Home', 'GF3Away', 'GA3Home', 'GA3Away', 'GF5Home', 'GF5Away', 'GA5Home', 'GA5Away',
        'OddHome', 'OddDraw', 'OddAway', 'ImpliedProbHome', 'ImpliedProbDraw', 'ImpliedProbAway', 'BookmakerMargin',
        'HandiSize', 'HandiHome', 'HandiAway', 
        'MaxHome', 'MaxDraw', 'MaxAway', 'Over25', 'Under25', 'MaxOver25', 'MaxUnder25', 
        'WinStreakHome', 'WinStreakAway', 'DrawStreakHome', 'DrawStreakAway', 'DefeatStreakHome', 'DefeatStreakAway', 'H2HHomeWins', 'H2HAwayWins', 'GF_EMA3_Home','GF_EMA3_Away', 'GF3HomeSTD',
        'GF3AwaySTD', 'Season', 'PointsAcumHome', 'PointsAcumAway', 
        'GF_Total_Home', 'GF_Total_Away', 'GA_Total_Home', 'GA_Total_Away', 
        'GD_total_Home', 'GD_total_Away', 'PointMeanHome', 'PointMeanAway', 'ScoredGoalsMeanHome', 'ScoredGoalsMeanAway', 'ConcededGoalsMeanHome', 'ConcededGoalsMeanAway',
        'GoalsDifferenceMeanHome','GoalsDifferenceMeanAway','WinHomeAcum','WinAwayAcum','DrawHomeAcum','DrawAwayAcum','LossHomeAcum', 'LossAwayAcum', 'WinRateHome','WinRateAway','DrawRateHome',
        'DrawRateAway','LossRateHome','LossRateAway', 
        "OddsDifference", "EloRatio", "FormRatio", "GoalRateRatio", "WinRateDiff", "PointsDiff", "FormDiff", "Elo_ProbDiff", "FormOddsDiff", "StreakDiff", "BookieBiasHome", "DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos", "Day_sin", "Day_cos",
        'OddSkew', 'FormVolatility', 'EloOddsGap']

# Numerical Colums without draw
num_cols_nd= ['Year',  'IsWeekend',
        'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB',
        'HomeElo', 'AwayElo', 'EloDifference', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'Form3Difference', 'Form5Difference',
        'GF3Home', 'GF3Away', 'GA3Home', 'GA3Away', 'GF5Home', 'GF5Away', 'GA5Home', 'GA5Away',
        'OddHome', 'OddAway', 'ImpliedProbHome', 'ImpliedProbAway', 'BookmakerMargin',
        'HandiSize', 'HandiHome', 'HandiAway', 
        'MaxHome', 'MaxAway', 'Over25', 'Under25', 'MaxOver25', 'MaxUnder25', 
        'WinStreakHome', 'WinStreakAway', 'DefeatStreakHome', 'DefeatStreakAway', 'H2HHomeWins', 'H2HAwayWins', 'GF_EMA3_Home','GF_EMA3_Away', 'GF3HomeSTD',
        'GF3AwaySTD', 'Season', 'PointsAcumHome', 'PointsAcumAway', 
        'GF_Total_Home', 'GF_Total_Away', 'GA_Total_Home', 'GA_Total_Away', 
        'GD_total_Home', 'GD_total_Away', 'PointMeanHome', 'PointMeanAway', 'ScoredGoalsMeanHome', 'ScoredGoalsMeanAway', 'ConcededGoalsMeanHome', 'ConcededGoalsMeanAway',
        'GoalsDifferenceMeanHome','GoalsDifferenceMeanAway','WinHomeAcum','WinAwayAcum','LossHomeAcum', 'LossAwayAcum', 'WinRateHome','WinRateAway',
        'LossRateHome','LossRateAway', 
        "OddsDifference", "EloRatio", "FormRatio", "GoalRateRatio", "WinRateDiff", "PointsDiff", "FormDiff", "Elo_ProbDiff", "FormOddsDiff", "StreakDiff", "BookieBiasHome", "DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos", "Day_sin", "Day_cos",
        'OddSkew', 'FormVolatility', 'EloOddsGap']



num_cols_all1 =  num_cols_nd   
X_train_nd[num_cols_all1] = X_train_nd[num_cols_all1].fillna(X_train_nd[num_cols_all1].mean())
X_test_nd[num_cols_all1] = X_test_nd[num_cols_all1].fillna(X_train_nd[num_cols_all1].mean())

num_pipe = Pipeline(steps=[ 
('imputer', SimpleImputer(strategy='mean')),
('scaler', StandardScaler())
])


# Categorical pipe with OneHot
cat_pipe_onehot = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Categorical pipe with Ordinal
cat_pipe_ordinal = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])


onehot_transformer = ColumnTransformer(transformers= [
    ('num_pipeline', num_pipe, num_cols),
    ('cat_pipeline', cat_pipe_onehot, cat_cols)], 
    remainder= 'passthrough',
    n_jobs= -1)

onehot_transformer_nd = ColumnTransformer(transformers= [
    ('num_pipeline', num_pipe, num_cols_nd),
    ('cat_pipeline', cat_pipe_onehot, cat_cols)], 
    remainder= 'passthrough',
    n_jobs= -1)


ordinal_transformer = ColumnTransformer(transformers= [
    ('num_pipeline', num_pipe, num_cols),
    ('cat_pipeline', cat_pipe_ordinal, cat_cols)], 
    remainder= 'passthrough',
    n_jobs= -1)

ordinal_transformer_nd = ColumnTransformer(transformers= [
    ('num_pipeline', num_pipe, num_cols_nd),
    ('cat_pipeline', cat_pipe_ordinal, cat_cols)], 
    remainder= 'passthrough',
    n_jobs= -1)

pipe_rf_nd = Pipeline(steps=[
('preprocessor', onehot_transformer_nd),
('classifier', RandomForestClassifier(random_state=42))
])

random_forest_params_nd = {
'preprocessor__num_pipeline__scaler': [StandardScaler(), MinMaxScaler(), None],
'preprocessor': [onehot_transformer_nd, ordinal_transformer_nd],
"classifier__n_estimators": [200, 300, 500],
'classifier__min_samples_split': [2, 5, 10],
'classifier__min_samples_leaf': [1, 2, 4],
'classifier__max_features': ['sqrt', 'log2'],
"classifier__max_depth": [5, 10, 15]
}

cv = TimeSeriesSplit(n_splits=5)
random_forest_RS_nd = RandomizedSearchCV(
pipe_rf_nd,
param_distributions=random_forest_params_nd,
cv=TimeSeriesSplit(n_splits=5),
scoring="f1",
n_jobs=1,
verbose=2,
random_state=42
)
random_forest_RS_nd.fit(X_train_nd, y_train_nd)


# Avaliação
y_pred = random_forest_RS_nd.predict(X_test_nd)
print("Accuracy:", accuracy_score(y_test_nd, y_pred))
print(confusion_matrix(y_test_nd, y_pred))
print(classification_report(y_test_nd, y_pred))

# Probabilidades
y_proba = random_forest_RS_nd.predict_proba(X_test_nd)
print("Classes:", random_forest_RS_nd.classes_)
print("Primeiras probabilidades:\n", y_proba[:5])
