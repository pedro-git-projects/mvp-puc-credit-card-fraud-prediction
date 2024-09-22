from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

knn_model_path = os.path.join(MODEL_DIR, "knn_optimized_pipeline.pkl")
decision_tree_model_path = os.path.join(MODEL_DIR, "dt_optimized_pipeline.pkl")
naive_bayes_model_path = os.path.join(MODEL_DIR, "nb_optimized_pipeline.pkl")

knn_model = joblib.load(knn_model_path)
decision_tree_model = joblib.load(decision_tree_model_path)
naive_bayes_model = joblib.load(naive_bayes_model_path)


class Transaction(BaseModel):
    time_elapsed: int
    amt: float
    lat: float
    long: float


@app.get("/")
def read_root():
    return {"message": "API de Detecção de Fraude está funcionando!"}


@app.post("/predict-knn")
def predict_knn(transaction: Transaction):
    expected_columns = ['lat', 'long', 'time_elapsed', 'amt']
    
    input_data = pd.DataFrame([transaction.model_dump()])[expected_columns]
    
    logger.info(f"Columns in input data: {list(input_data.columns)}")
    logger.info(f"Input data values: {input_data.values}")
    
    prediction = knn_model.predict(input_data)
    
    logger.info(f"KNN Prediction: {prediction[0]}")
    
    return {"prediction": int(prediction[0])}
@app.post("/predict-decision-tree")
def predict_decision_tree(transaction: Transaction):
    expected_columns = ['lat', 'long', 'time_elapsed', 'amt']

    input_data = pd.DataFrame([transaction.model_dump()])[expected_columns]
    prediction = decision_tree_model.predict(input_data)
    return {"prediction": int(prediction[0])}


@app.post("/predict-naive-bayes")
def predict_naive_bayes(transaction: Transaction):
    expected_columns = ['lat', 'long', 'time_elapsed', 'amt']
    input_data = pd.DataFrame([transaction.model_dump()])[expected_columns]
    prediction = naive_bayes_model.predict(input_data)
    return {"prediction": int(prediction[0])}
