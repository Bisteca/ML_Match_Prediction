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


def training():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_file = os.path.join(base_dir, '..', 'data', 'processed', 'matches_processed.csv')
    df = pd.read_csv(raw_file)


    df = df[df['MatchDate'] > '2010-08']


    counts_home = df['HomeTeam'].value_counts()
    counts_away = df['AwayTeam'].value_counts()

    valid_teams = counts_home[counts_home >= 20].index.intersection(counts_away[counts_away >= 20].index)
    df = df[(df['HomeTeam'].isin(valid_teams)) & (df['AwayTeam'].isin(valid_teams))]


    X = df[['MatchDate', 'HomeTeam', 'AwayTeam', 'Year',  'IsWeekend',
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




    X_train = X[X['MatchDate'] < '2021-07'].drop(columns='MatchDate')
    X_test = X[X['MatchDate'] > '2021-08'].drop(columns='MatchDate')
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]



    # X and y without draw
    X_train_nd = X_not_draw[X_not_draw['MatchDate'] < '2021-07'].drop(columns='MatchDate')
    X_test_nd = X_not_draw[X_not_draw['MatchDate'] > '2021-08'].drop(columns='MatchDate')
    y_train_nd = y_not_draw.loc[X_train_nd.index]
    y_test_nd = y_not_draw.loc[X_test_nd.index]



    os.makedirs('../data/train', exist_ok=True)
    os.makedirs('../data/test', exist_ok=True)

    X_train.to_csv('../data/train/X_train.csv', index=False)
    X_test.to_csv('../data/test/X_test.csv', index=False)
    y_train.to_csv('../data/train/y_train.csv', index=False)
    y_test.to_csv('../data/test/y_test.csv', index=False)

    X_train_nd.to_csv('../data/train/X_train_nd.csv', index=False)
    X_test_nd.to_csv('../data/test/X_test_nd.csv', index=False)
    y_train_nd.to_csv('../data/train/y_train_nd.csv', index=False)
    y_test_nd.to_csv('../data/test/y_test_nd.csv', index=False)


    # Categorical colums
    cat_cols = ['HomeTeam', 'AwayTeam']



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
    


    num_pipe = Pipeline(steps=[ 
    ('scaler', StandardScaler())
    ])


    # Categorical pipe with OneHot
    cat_pipe_onehot = Pipeline(steps=[
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Categorical pipe with Ordinal
    cat_pipe_ordinal = Pipeline(steps=[
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


    pipe_LR = Pipeline(steps=[
                ('preprocessor', onehot_transformer),
                ('smote', SMOTE(random_state=42)),
                ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))])

    logistic_params = {
            'preprocessor': [onehot_transformer, ordinal_transformer],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__class_weight': [None, 'balanced']}

    logistic_RS = RandomizedSearchCV(estimator = pipe_LR,
                            param_distributions=  logistic_params,
                            cv = 5,
                            verbose=2,
                            n_jobs=1, 
                            scoring= 'accuracy',
                            random_state=42)

    logistic_RS.fit(X_train, y_train)
    y_pred_logistic = logistic_RS.predict(X_test)



    pipe_LR_nd = Pipeline(steps=[
            ('preprocessor', onehot_transformer_nd),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))])

    logistic_params_nd = {
            'preprocessor': [onehot_transformer_nd, ordinal_transformer_nd],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__class_weight': [None, 'balanced']}

    logistic_RS_nd = RandomizedSearchCV(estimator = pipe_LR_nd,
                            param_distributions=  logistic_params_nd,
                            cv = 5,
                            verbose=2,
                            n_jobs=1, 
                            scoring= 'accuracy',
                            random_state=42)
    
    logistic_RS_nd.fit(X_train_nd, y_train_nd)
    y_pred_logistic_nd = logistic_RS_nd.predict(X_test_nd)


    pipe_rf = Pipeline(steps=[
            ('preprocessor', onehot_transformer),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))])

    random_forest_params = {
        'preprocessor__num_pipeline__scaler': [StandardScaler(), MinMaxScaler(), None],
        'preprocessor': [onehot_transformer, ordinal_transformer],
        "classifier__n_estimators": [200, 300, 500, 1000],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
        "classifier__max_depth": [3, 5, 7, 15]}

    random_forest_RS = RandomizedSearchCV(estimator = pipe_rf,
                        param_distributions=  random_forest_params,
                        cv = 5,
                        verbose=2,
                        n_jobs=1, 
                        scoring= 'accuracy',
                        random_state=42)

    random_forest_RS.fit(X_train, y_train)
    y_pred_rf = random_forest_RS.predict(X_test)


    pipe_rf_nd = Pipeline(steps=[
            ('preprocessor', onehot_transformer_nd),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))])

    random_forest_params_nd = {
        'preprocessor__num_pipeline__scaler': [StandardScaler(), MinMaxScaler(), None],
        'preprocessor': [onehot_transformer_nd, ordinal_transformer_nd],
        "classifier__n_estimators": [200, 300, 500, 1000],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
        "classifier__max_depth": [3, 5, 7, 15]}

    random_forest_RS_nd = RandomizedSearchCV(estimator = pipe_rf_nd,
                        param_distributions=  random_forest_params_nd,
                        cv = 5,
                        verbose=2,
                        n_jobs=1, 
                        scoring= 'accuracy',
                        random_state=42)

    random_forest_RS_nd.fit(X_train_nd, y_train_nd)
    y_pred_rf_nd = random_forest_RS_nd.predict(X_test_nd)



    pipe_xgb = Pipeline(steps=[
            ('preprocessor', onehot_transformer),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(random_state=42))])
    xgb_params = {
        'preprocessor': [onehot_transformer, ordinal_transformer],
        "classifier__n_estimators": [200, 500],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8, 1],
        "classifier__colsample_bytree": [0.8, 1]
    }

    xgb_RS = RandomizedSearchCV(estimator = pipe_xgb,
                        param_distributions=  xgb_params,
                        cv = 5,
                        verbose=2,
                        n_jobs=1, 
                        scoring= 'f1_macro',
                        random_state=42)

    xgb_RS.fit(X_train, y_train)
    y_pred_xgb = xgb_RS.predict(X_test)


    pipe_xgb_nd = Pipeline(steps=[
            ('preprocessor', onehot_transformer_nd),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(random_state=42))])

    xgb_params_nd = {
        'preprocessor': [onehot_transformer_nd, ordinal_transformer_nd],
        "classifier__n_estimators": [200, 500],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8, 1],
        "classifier__colsample_bytree": [0.8, 1]}

    xgb_RS_nd = RandomizedSearchCV(estimator = pipe_xgb_nd,
                        param_distributions=  xgb_params_nd,
                        cv = 5,
                        verbose=2,
                        n_jobs=1, 
                        scoring= 'accuracy',
                        random_state=42)


    y_train_xgb = y_train_nd.map({0: 0, 2: 1})
    y_test_xgb = y_test_nd.map({0: 0, 2: 1})

    xgb_RS_nd.fit(X_train_nd, y_train_xgb)
    y_pred_xgb_nd = xgb_RS_nd.predict(X_test_nd)


    pipe_knn = Pipeline(steps=[
            ('preprocessor', onehot_transformer),
            ('smote', SMOTE(random_state=42)),
            ('classifier', KNeighborsClassifier())])

    knn_params = {
        'preprocessor': [onehot_transformer, ordinal_transformer],
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]  }

    knn_RS = RandomizedSearchCV(estimator = pipe_knn,
                        param_distributions=  knn_params,
                        cv = 5,
                        verbose=2,
                        n_jobs=1, 
                        scoring= 'accuracy',
                        random_state=42)

    knn_RS.fit(X_train, y_train)
    y_pred_knn = knn_RS.predict(X_test)


    pipe_knn_nd = Pipeline(steps=[
            ('preprocessor', onehot_transformer_nd),
            ('smote', SMOTE(random_state=42)),
            ('classifier', KNeighborsClassifier())
            ])

    knn_params_nd = {
        'preprocessor': [onehot_transformer_nd, ordinal_transformer_nd],
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]  
    }

    knn_RS_nd = RandomizedSearchCV(estimator = pipe_knn_nd,
                        param_distributions=  knn_params_nd,
                        cv = 5,
                        verbose=2,
                        n_jobs=1, 
                        scoring= 'accuracy',
                        random_state=42)


    knn_RS_nd.fit(X_train_nd, y_train_nd)
    y_pred_knn_nd = knn_RS_nd.predict(X_test_nd)


    pipe_gbm = Pipeline(steps=[
            ('preprocessor', onehot_transformer),
            ('smote', SMOTE(random_state=42)),
            ('classifier', GradientBoostingClassifier(random_state=42))
            ])

    gbm_params = {
        'preprocessor': [onehot_transformer, ordinal_transformer],
        'classifier__n_estimators': [100, 200, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]}

    gbm_RS = RandomizedSearchCV(estimator = pipe_gbm,
                        param_distributions=  gbm_params,
                        cv = 5,
                        verbose=2,
                        n_jobs=1, 
                        scoring= 'accuracy',
                        random_state=42)


    gbm_RS.fit(X_train, y_train)
    y_pred_gbm = gbm_RS.predict(X_test)



    pipe_gbm_nd = Pipeline(steps=[
            ('preprocessor', onehot_transformer_nd),
            ('smote', SMOTE(random_state=42)),
            ('classifier', GradientBoostingClassifier(random_state=42))])

    gbm_params_nd = {
        'preprocessor': [onehot_transformer_nd, ordinal_transformer_nd],
        'classifier__n_estimators': [100, 200, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]}

    gbm_RS_nd = RandomizedSearchCV(estimator = pipe_gbm_nd,
                        param_distributions=  gbm_params_nd,
                        cv = 5,
                        verbose=2,
                        n_jobs=1, 
                        scoring= 'accuracy',
                        random_state=42)

    gbm_RS_nd.fit(X_train_nd, y_train_nd)
    y_pred_gbm_nd = gbm_RS_nd.predict(X_test_nd)


    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    trained_models = [
        ('LogisticRegression_NoDraw', logistic_RS_nd),
        ('LogisticRegression_Draw', logistic_RS),
        ('RandomForest_NoDraw', random_forest_RS_nd),
        ('RandomForest_Draw', random_forest_RS),
        ('XGB_NoDraw', xgb_RS_nd),
        ('XGB_Draw', xgb_RS),
        ('KNN_NoDraw', knn_RS_nd),
        ('KNN_Draw', knn_RS),
        ('GBM_NoDraw', gbm_RS_nd),
        ('GBM_Draw', gbm_RS),

            
    ]

    for name, model in trained_models:
        file_path = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(model, file_path)
        print(f"Saved {name} at {file_path}")

    final_model_path = os.path.join(models_dir, 'final_model.joblib')
    joblib.dump(random_forest_RS_nd, final_model_path)
    print(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    training()
