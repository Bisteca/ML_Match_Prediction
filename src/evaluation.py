import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def evaluate_models():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test'))
    X_test = pd.read_csv(os.path.join(base_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(base_dir, 'y_test.csv')).squeeze()  

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"Erro ao carregar {model_file}: {e}")
            continue

        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Erro ao prever com {model_file}: {e}")
            continue

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        print(f'== Modelo: {model_file} ==')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print('\n' + classification_report(y_test, y_pred))
        print('---------------------------------------')


if __name__ == "__main__":
    evaluate_models()
