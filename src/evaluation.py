import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def evaluate_models():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test'))
    
    X_test = pd.read_csv(os.path.join(test_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(test_dir, 'y_test.csv')).squeeze()  

    X_test_nd = pd.read_csv(os.path.join(test_dir, 'X_test_nd.csv'))
    y_test_nd = pd.read_csv(os.path.join(test_dir, 'y_test_nd.csv')).squeeze()

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]

    results = []

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(f'\n== Avaliando modelo: {model_file} ==')

        try:
            model = joblib.load(model_path)
        except ModuleNotFoundError as mnfe:
            print(f"Erro: Módulo faltando ao carregar {model_file}: {mnfe}")
            continue
        except Exception as e:
            print(f"Erro ao carregar {model_file}: {e}")
            continue

        # Detecta se é "no draw" pelo nome do arquivo
        if "_NoDraw" or "final_model" in model_file:
            X_eval = X_test_nd
            y_eval = y_test_nd

        else:
            X_eval = X_test
            y_eval = y_test



        if "_NoDraw" in model_file:
        # Remove empates
            y_eval = y_test_nd[y_test_nd != 1]
            X_eval = X_test_nd.loc[y_eval.index]

        # Se for XGB, aplica mapeamento binário
        if "XGB" in model_file:
            y_eval = y_eval.map({0: 0, 2: 1})

        try:
            y_pred = model.predict(X_eval)
        except Exception as e:
            print(f"Erro ao prever com {model_file}: {e}")
            continue

        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average='macro')
        precision = precision_score(y_eval, y_pred, average='macro')
        recall = recall_score(y_eval, y_pred, average='macro')

        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print('\n' + classification_report(y_eval, y_pred))
        print('---------------------------------------')

        results.append({
            "model": model_file,
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall
        })

    # Opcional: salvar relatório completo em CSV
    results_df = pd.DataFrame(results)
    report_path = os.path.join(models_dir, 'evaluation_report.csv')
    results_df.to_csv(report_path, index=False)
    print(f'\n✅ Evaluation report saved at {report_path}')


if __name__ == "__main__":
    evaluate_models()
