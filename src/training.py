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


def training():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_file = os.path.join(base_dir, '..', 'data', 'processed', 'processed_data.csv')
    df = pd.read_csv(raw_file)

    ligas = ['SP1','E0','D1','I1','F1']
    modelos = {}
    X_not_draw_dict = {}
    df = df[df['MatchDate'] > '2010-08']


    counts_home = df['HomeTeam'].value_counts()
    counts_away = df['AwayTeam'].value_counts()

    valid_teams = counts_home[counts_home >= 20].index.intersection(counts_away[counts_away >= 20].index)
    df = df[(df['HomeTeam'].isin(valid_teams)) & (df['AwayTeam'].isin(valid_teams))]

    for liga in ligas:
        df_liga = df[df['Division'] == liga].copy()

        X = df_liga[['Division','MatchDate', 'HomeTeam', 'AwayTeam', 'Year',  'IsWeekend',
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
        y = df_liga['Results']

        # elimina empates
        cols_draw = ['DrawStreakHome', 'DrawStreakAway', 
                    'DrawHomeAcum', 'DrawAwayAcum',
                    'DrawRateHome', 'DrawRateAway',
                    'OddDraw', 'ImpliedProbDraw', 'MaxDraw']
        # Dropping Draws
        not_draw= y != 1
        not_draw= y != 1
        X_not_draw = X[not_draw].drop(columns=cols_draw).copy()
        y_not_draw = y[not_draw].copy()

        X_history = X_not_draw.copy() 
        if 'Division' not in X_history.columns:
            X_history['Division'] = df['Division'] 

        os.makedirs('../data/processed', exist_ok=True)
        X_history.to_csv('../data/processed/X_history.csv', index=False)

        X_train = X[X['MatchDate'] < '2021-07'].drop(columns='MatchDate')
        X_test = X[X['MatchDate'] > '2021-08'].drop(columns='MatchDate')
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]



        # X and y without draw
        X_train_nd = X_not_draw[X_not_draw['MatchDate'] < '2021-07'].drop(columns='MatchDate')
        X_test_nd = X_not_draw[X_not_draw['MatchDate'] > '2021-08'].drop(columns='MatchDate')
        y_train_nd = y_not_draw.loc[X_train_nd.index]
        y_test_nd = y_not_draw.loc[X_test_nd.index]

        for X_ in [X_train_nd, X_test_nd]:
            X_.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_.fillna(0, inplace=True)

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
        

        num_cols_all = num_cols   
        X_train[num_cols_all] = X_train[num_cols_all].fillna(X_train[num_cols_all].mean())
        X_test[num_cols_all] = X_test[num_cols_all].fillna(X_train[num_cols_all].mean())

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

        #logistic_RS.fit(X_train, y_train)



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
        
        #logistic_RS_nd.fit(X_train_nd, y_train_nd)


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

        #random_forest_RS.fit(X_train, y_train)


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
                            n_jobs=-1, 
                            scoring= 'accuracy',
                            random_state=42)

        random_forest_RS_nd.fit(X_train_nd, y_train_nd)



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

        #xgb_RS.fit(X_train, y_train)


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

        #xgb_RS_nd.fit(X_train_nd, y_train_xgb)


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

        #knn_RS.fit(X_train, y_train)


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


        #knn_RS_nd.fit(X_train_nd, y_train_nd)


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


        #gbm_RS.fit(X_train, y_train)



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

        #gbm_RS_nd.fit(X_train_nd, y_train_nd)


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
        
        modelos[liga] = random_forest_RS_nd
        X_not_draw_dict[liga] = X_not_draw

        models_dir = os.path.join(base_dir, '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"RandomForest_NoDraw_{liga}.joblib")
        joblib.dump(random_forest_RS_nd, model_path)
        print(f"Saved RandomForest_NoDraw for {liga} at {model_path}")


        processed_dir = os.path.join(base_dir, '..', 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        xnd_path = os.path.join(processed_dir, f"X_not_draw_{liga}.csv")
        X_not_draw.to_csv(xnd_path, index=False)
        print(f"Saved X_not_draw for {liga} at {xnd_path}")

if __name__ == "__main__":
    training()
