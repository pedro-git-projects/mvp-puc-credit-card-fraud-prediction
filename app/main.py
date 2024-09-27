from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import knn_router, naive_bayes_router, decision_tree_router
from .logging_config import logging, logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(knn_router, prefix="/predict-knn", tags=["KNN"])
app.include_router(
    decision_tree_router, prefix="/predict-decision-tree", tags=["Decision Tree"]
)
app.include_router(
    naive_bayes_router, prefix="/predict-naive-bayes", tags=["Naive Bayes"]
)


@app.get("/")
def read_root():
    return {"message": "API de Detecção de Fraude está funcionando!"}
