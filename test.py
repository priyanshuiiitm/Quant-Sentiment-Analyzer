import pandas as pd
from src.data_pipeline import fetch_ohlcv,add_technical_indicators,save_processed_market_data
df=fetch_ohlcv(ticker="TCS.NS",start="2022-01-01",end="2025-01-01",save_path="Data/raw/market/TCS.csv")
df=add_technical_indicators(df)
save_processed_market_data(df,"TCS")
pd.read_parquet("data/processed/market/TCS.parquet").tail()
