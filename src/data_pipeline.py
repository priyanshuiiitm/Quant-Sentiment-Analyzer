import os
from pathlib import Path
import certifi
path_to_cert=certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = path_to_cert
os.environ['SSL_CERT_FILE'] = path_to_cert
import yfinance as yf
yf_cache_dir=Path("Data/cache/yfinance").resolve()
yf_cache_dir.mkdir(parents=True,exist_ok=True)
yf.set_tz_cache_location(str(yf_cache_dir))
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
    df=yf.download(ticker,start=start,end=end,progress=False,auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    df=df.reset_index()
    Path(save_path).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(save_path,index=False)
    return df
def add_technical_indicators(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    if df.empty or "Close" not in df.columns:
        return df
    close=df["Close"]
    if isinstance(close,pd.DataFrame):
        close=close.iloc[:,0]
    close=pd.to_numeric(close,errors="coerce")
    if close.dropna().empty:
        return df
    df["Close"]=close
    df["log_return"]=np.log(df["Close"]/df["Close"].shift(1))
    df["rsi"]=ta.momentum.RSIIndicator(close=df["Close"],window=14).rsi()
    macd=ta.trend.MACD(close=df["Close"])
    df["macd"]=macd.macd()
    df["macd_signal"]=macd.macd_signal()
    df["sma_20"]=df["Close"].rolling(window=20).mean()
    df["sma_50"]=df["Close"].rolling(window=50).mean()
    df["volatility_20"]=df["log_return"].rolling(window=20).std()
    return df
def save_processed_market_data(df:pd.DataFrame,company:str):
    path=f"Data/processed/market/{company}.parquet"
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(path)
