from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

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
decision_tree_model_path = os.path.join(MODEL_DIR, "decision_tree_pipeline.pkl")
naive_bayes_model_path = os.path.join(MODEL_DIR, "naive_bayes_pipeline.pkl")
svm_model_path = os.path.join(MODEL_DIR, "svm_pipeline.pkl")

knn_model = joblib.load(knn_model_path)
decision_tree_model = joblib.load(decision_tree_model_path)
naive_bayes_model = joblib.load(naive_bayes_model_path)
svm_model = joblib.load(svm_model_path)


class Transaction(BaseModel):
    index: int
    time_elapsed: int
    cc_num: float
    amt: float
    lat: float
    long: float


@app.get("/")
def read_root():
    return {"message": "API de Detecção de Fraude está funcionando!"}


@app.post("/predict-knn")
def predict_knn(transaction: Transaction):
    input_data = pd.DataFrame([transaction.model_dump()])
    prediction = knn_model.predict(input_data)
    return {"prediction": int(prediction[0])}


@app.post("/predict-decision-tree")
def predict_decision_tree(transaction: Transaction):
    input_data = pd.DataFrame([transaction.model_dump()])
    prediction = decision_tree_model.predict(input_data)
    return {"prediction": int(prediction[0])}


@app.post("/predict-naive-bayes")
def predict_naive_bayes(transaction: Transaction):
    input_data = pd.DataFrame([transaction.model_dump()])
    prediction = naive_bayes_model.predict(input_data)
    return {"prediction": int(prediction[0])}
