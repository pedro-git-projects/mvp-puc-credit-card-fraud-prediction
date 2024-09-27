import joblib
import pandas as pd
import logging
from .config import MODEL_DIR
import os

logger = logging.getLogger(__name__)

knn_model_path = os.path.join(MODEL_DIR, "knn_optimized_pipeline.pkl")
decision_tree_model_path = os.path.join(MODEL_DIR, "dt_optimized_pipeline.pkl")
naive_bayes_model_path = os.path.join(MODEL_DIR, "nb_optimized_pipeline.pkl")

knn_model = joblib.load(knn_model_path)
decision_tree_model = joblib.load(decision_tree_model_path)
naive_bayes_model = joblib.load(naive_bayes_model_path)


def predict(model, transaction_data):
    expected_columns = ["lat", "long", "time_elapsed", "amt"]
    input_data = pd.DataFrame([transaction_data])[expected_columns]

    logger.info(f"Input data columns: {input_data.columns}")
    logger.info(f"Input data values: {input_data.values}")

    prediction = model.predict(input_data)
    return int(prediction[0])
