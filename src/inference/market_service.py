import yfinance as yf
import pandas as pd
import ta
import numpy as np
import os
from pathlib import Path
import certifi
path_to_cert=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=path_to_cert
os.environ["SSL_CERT_FILE"]=path_to_cert
yf_cache_dir=Path("Data/cache/yfinance").resolve()
yf_cache_dir.mkdir(parents=True,exist_ok=True)
yf.set_tz_cache_location(str(yf_cache_dir))
def get_market_features(ticker):
    df=yf.download(ticker,period="180d",progress=False,auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No market data returned for {ticker}")
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    close=pd.to_numeric(df["Close"]/df["Close"].shift(1))
    df["Close"]=close
    df["log_return"]=np.log(df["Close"]/df["Close"].shift(1))
    df["rsi"]=ta.momentum.RSIIndicator(df["Close"],window=14).rsi()
    macd=ta.trend.MACD(df["Close"])
    df["macd"]=macd.macd()
    df["macd_signal"]=macd.macd_signal()
    df["sma_20"]=df["Close"].rolling(20).mean()
    df["sma_50"]=df["Close"].rolling(50).mean()
    df["volatility_20"]=df["log_return"].rolling(20).std()
    feature_cols=[
        "Open","High","Low","Close","Volume","log_return","rsi","macd","macd_signal","sma_20","sma_50","volatility_20"
    ]
    window=df[feature_cols].dropna().tail(60)
    if len(window)<60:
        raise ValueError(
            f"Not enough market data for {ticker}. Got {len(window)} rows, need 60."
        )
    return window.values.astype(np.float32)