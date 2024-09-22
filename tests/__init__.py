import pytest
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Definir os thresholds
MIN_PRECISION = 0.90
MIN_RECALL = 0.90
MIN_F1_SCORE = 0.90

# Caminho base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminho para o modelo salvo
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'knn_optimized_pipeline.pkl')

# Caminho para o conjunto de teste
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test_data.csv')

@pytest.fixture
def load_model():
    """Carrega o modelo salvo."""
    model = joblib.load(MODEL_PATH)
    return model

@pytest.fixture
def load_test_data():
    """Carrega o conjunto de dados de teste."""
    data = pd.read_csv(TEST_DATA_PATH)
    X_test = data[['lat', 'long', 'time_elapsed', 'amt']]
    y_test = data['is_fraud']
    return X_test, y_test

def test_model_performance(load_model, load_test_data):
    model = load_model
    X_test, y_test = load_test_data

    # Fazer predições
    y_pred = model.predict(X_test)

    # Calcular métricas
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Imprimir métricas para referência
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Verificar se as métricas atendem aos thresholds
    assert precision >= MIN_PRECISION, f"Precisão abaixo do mínimo: {precision:.2f} < {MIN_PRECISION}"
    assert recall >= MIN_RECALL, f"Recall abaixo do mínimo: {recall:.2f} < {MIN_RECALL}"
    assert f1 >= MIN_F1_SCORE, f"F1-Score abaixo do mínimo: {f1:.2f} < {MIN_F1_SCORE}"
