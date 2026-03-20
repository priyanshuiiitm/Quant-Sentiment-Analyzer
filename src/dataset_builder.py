import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.companies import COMPANIES
def load_market_data(company:str)->pd.DataFrame:
    path=f"Data/processed/market/{company}.parquet"
    df=pd.read_parquet(path)
    df["Date"]=pd.to_datetime(df["Date"])
    return df
def load_news_data()->pd.DataFrame:
    path="Data/processed/news/daily_news_embeddings.parquet"
    df=pd.read_parquet(path)
    df["date"]=pd.to_datetime(df["date"])
    return df
def merge_market_and_news(market_df:pd.DataFrame,news_df:pd.DataFrame,company:str)->pd.DataFrame:
    company_news=news_df[news_df["company"]==company].drop(columns=["company"]).copy()
    df=market_df.merge(
        company_news,
        left_on="Date",
        right_on="date",
        how="left"
    )
    df["date"]=df["Date"]
    embedding_dim=len(news_df.iloc[0]["news_embedding"])
    df["has_news"]=df["has_news"].fillna(0).astype(int)
    df["news_volume"]=df["news_volume"].fillna(0).astype(int)
    df["news_embedding"]=df["news_embedding"].apply(
        lambda x:np.array(x) if isinstance(x,(list,np.ndarray)) else np.zeros(embedding_dim)
    )
    return df
def create_labels(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    df["label"]=(df["Close"].shift(-1)>df["Close"]).astype(int)
    return df.iloc[:-1]
def create_rolling_samples(df:pd.DataFrame,window:int=60):
    samples=[]
    feature_cols=[
        "Open","High","Low","Close","Volume","log_return","rsi","macd","macd_signal","sma_20","sma_50","volatility_20"]
    df=df.copy()
    df=df.dropna(subset=feature_cols).reset_index(drop=True)
    for i in range(window-1,len(df)):
        market_window=df.iloc[i-window+1:i+1][feature_cols].values
        samples.append({
            "company":df.iloc[i]["company"],
            "date":df.iloc[i]["date"],
            "market_window":market_window,
            "news_embedding":df.iloc[i]["news_embedding"],
            "news_volume":df.iloc[i]["news_volume"],
            "has_news":df.iloc[i]["has_news"],
            "close_price":df.iloc[i]["Close"],
            "label":df.iloc[i]["label"]
        })
    return pd.DataFrame(samples)
def time_split(df:pd.DataFrame):
    n=len(df)
    train_end=int(0.7*n)
    val_end=int(0.85*n)
    train=df.iloc[:train_end]
    val=df.iloc[train_end:val_end]
    test=df.iloc[val_end:]
    return train,val,test
def normalize_market_windows(train,val,test):
    scaler=StandardScaler()
    train_windows=np.vstack(train["market_window"].values)
    scaler.fit(train_windows)
    def transform(df):
        windows=df["market_window"].values
        df["market_window"]=[scaler.transform(w) for w in windows]
        return df
    return transform(train),transform(val),transform(test)
def build_final_dataset():
    news_df=load_news_data()
    all_train,all_val,all_test=[],[],[]
    for company in COMPANIES.keys():
        market_df=load_market_data(company)
        market_df["company"]=company
        df=merge_market_and_news(market_df,news_df,company)
        df=create_labels(df)
        samples=create_rolling_samples(df)
        train,val,test=time_split(samples)
        train,val,test=normalize_market_windows(train,val,test)
        all_train.append(train)
        all_val.append(val)
        all_test.append(test)
    train_df=pd.concat(all_train).reset_index(drop=True)
    val_df=pd.concat(all_val).reset_index(drop=True)
    test_df=pd.concat(all_test).reset_index(drop=True)
    Path("Data/processed/final").mkdir(parents=True,exist_ok=True)
    train_df["market_window"]=train_df["market_window"].apply(lambda x:x.tolist() if isinstance(x,np.ndarray) else x)
    val_df["market_window"]=val_df["market_window"].apply(lambda x:x.tolist() if isinstance(x,np.ndarray) else x)
    test_df["market_window"]=test_df["market_window"].apply(lambda x:x.tolist() if isinstance(x,np.ndarray) else x)
    train_df["news_embedding"]=train_df["news_embedding"].apply(lambda x:x.tolist() if isinstance(x,np.ndarray) else x)
    val_df["news_embedding"]=val_df["news_embedding"].apply(lambda x:x.tolist() if isinstance(x,np.ndarray) else x)
    test_df["news_embedding"]=test_df["news_embedding"].apply(lambda x:x.tolist() if isinstance(x,np.ndarray) else x)
    train_df.to_parquet("Data/processed/final/train.parquet")
    val_df.to_parquet("Data/processed/final/val.parquet")
    test_df.to_parquet("Data/processed/final/test.parquet")
    print("stage 4 completed")