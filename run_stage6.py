import torch 
import pandas as pd
from src.inference.predictor import load_model,predict_dataset
from src.backtesting.backtester import run_backtest
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=load_model("model.pth",device)
test_df=pd.read_parquet("Data/processed/final/test.parquet")
test_df=predict_dataset(model,test_df,device)
capital_curve=run_backtest(test_df)