from fastapi import APIRouter
from .models import Transaction
from .predictors import predict, knn_model, decision_tree_model, naive_bayes_model

knn_router = APIRouter()
decision_tree_router = APIRouter()
naive_bayes_router = APIRouter()


@knn_router.post("/")
def predict_knn(transaction: Transaction):
    prediction = predict(knn_model, transaction.model_dump())
    return {"prediction": prediction}


@decision_tree_router.post("/")
def predict_decision_tree(transaction: Transaction):
    prediction = predict(decision_tree_model, transaction.model_dump())
    return {"prediction": prediction}


@naive_bayes_router.post("/")
def predict_naive_bayes(transaction: Transaction):
    prediction = predict(naive_bayes_model, transaction.model_dump())
    return {"prediction": prediction}
