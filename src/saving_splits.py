import pandas as pd
import numpy as np
import os

def save_splits():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_file = os.path.join(base_dir, '..', 'data', 'processed', 'processed_data.csv')

    df = pd.read_csv(processed_file)

    # --- Mesmo pré-processamento que no training.py ---
    df = df[df['MatchDate'] > '2010-08']

    counts_home = df['HomeTeam'].value_counts()
    counts_away = df['AwayTeam'].value_counts()
    valid_teams = counts_home[counts_home >= 20].index.intersection(counts_away[counts_away >= 20].index)
    df = df[(df['HomeTeam'].isin(valid_teams)) & (df['AwayTeam'].isin(valid_teams))]

    # Features e target
    X = df.drop(columns=["Results"])
    y = df["Results"]

    # ----------------------------
    # Splits com empates
    # ----------------------------
    X_train = X[X['MatchDate'] < '2021-07']
    X_test = X[X['MatchDate'] > '2021-08']
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    # ----------------------------
    # Splits sem empates
    # ----------------------------
    cols_draw = ['DrawStreakHome', 'DrawStreakAway', 
                'DrawHomeAcum', 'DrawAwayAcum',
                'DrawRateHome', 'DrawRateAway',
                'OddDraw', 'ImpliedProbDraw', 'MaxDraw']

    not_draw = y != 1
    X_not_draw = X[not_draw].drop(columns=cols_draw)
    y_not_draw = y[not_draw]

    # Filtra os targets de treino e teste sem empate
    y_train_nd = y_not_draw[y_not_draw.index.isin(X[X['MatchDate'] < '2021-07'].index)]
    X_train_nd = X_not_draw.loc[y_train_nd.index]

    y_test_nd = y_not_draw[y_not_draw.index.isin(X[X['MatchDate'] > '2021-08'].index)]
    X_test_nd = X_not_draw.loc[y_test_nd.index]



    # ----------------------------
    # Criar pastas de destino
    # ----------------------------
    train_dir = os.path.join(base_dir, '..', 'data', 'train')
    test_dir = os.path.join(base_dir, '..', 'data', 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # ----------------------------
    # Salvar splits
    # ----------------------------
    X_train.to_csv(os.path.join(train_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(test_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(train_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(test_dir, 'y_test.csv'), index=False)

    X_train_nd.to_csv(os.path.join(train_dir, 'X_train_nd.csv'), index=False)
    X_test_nd.to_csv(os.path.join(test_dir, 'X_test_nd.csv'), index=False)
    y_train_nd.to_csv(os.path.join(train_dir, 'y_train_nd.csv'), index=False)
    y_test_nd.to_csv(os.path.join(test_dir, 'y_test_nd.csv'), index=False)

    # ----------------------------
    # Salvar datasets completos (úteis para Streamlit)
    # ----------------------------
    X.to_csv(os.path.join(train_dir, 'X_full.csv'), index=False)
    y.to_csv(os.path.join(train_dir, 'y_full.csv'), index=False)

    X_not_draw.to_csv(os.path.join(train_dir, 'X_not_draw_full.csv'), index=False)
    y_not_draw.to_csv(os.path.join(train_dir, 'y_not_draw_full.csv'), index=False)
    
    print("✅ All splits and full datasets saved successfully!")
    print("📂 Files saved in:", os.path.abspath(os.path.join(base_dir, '..', 'data')))


if __name__ == "__main__":
    save_splits()
