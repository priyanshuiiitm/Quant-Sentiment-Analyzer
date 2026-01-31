import os
path_to_cert=r"C:\Users\Priyanshu jha\OneDrive\文档\quant-sentiment-analyzer\venv\Lib\site-packages\certifi\cacert.pem"
os.environ['REQUESTS_CA_BUNDLE'] = path_to_cert
os.environ['SSL_CERT_FILE'] = path_to_cert
import yfinance as yf
from pathlib import Path
import pandas as pd
import numpy as np
import ta
def load_news_data(path:str)->pd.DataFrame:
    df=pd.read_csv(path)
    return df
def fetch_ohlcv(
        ticker:str,
        start:str,
        end:str,
        save_path:str
)->pd.DataFrame:
    df=yf.download(ticker,start=start,end=end,progress=False)
    df.reset_index(inplace=True)
    Path(save_path).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(save_path,index=False)
    return df
def add_technical_indicators(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    df["log_return"]=(df["Close"]/df["Close"].shift(1)).apply(lambda x:0 if pd.isna(x) else np.log(x))
    df["rsi"]=ta.momentum.RSIIndicator(close=df["Close"],window=14).rsi()
    macd=ta.trend.MACD(close=df["Close"])
    df["macd"]=macd.macd()
    df["sma_20"]=df["Close"].rolling(window=20).mean()
    df["sma_50"]=df["Close"].rolling(window=50).mean()
    df["volatility_20"]=df["log_return"].rolling(window=20).std()
    return df
def save_processed_market_data(df:pd.DataFrame,company:str):
    path=f"Data/processed/market/{company}.parquet"
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(path)
