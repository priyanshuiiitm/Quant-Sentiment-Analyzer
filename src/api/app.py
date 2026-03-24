from fastapi import FastAPI
from src.inference.inference_pipeline import predict_stock
app=FastAPI()
@app.get("/")
def root():
    return {"message":"Stock prediction API running"}
@app.get("/predict")
def predict(company:str,ticker:str):
    result=predict_stock(company,ticker)
    return result