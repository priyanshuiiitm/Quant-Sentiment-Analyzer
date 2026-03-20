import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from datetime import timedelta
from pathlib import Path
from src.companies import COMPANIES
def load_raw_news(path:str)->pd.DataFrame:
    df=pd.read_csv(path)
    df=df.rename(columns={
        "Date":"date",
        "Headline":"headline"
    })
    df=df[["date","headline"]]
    df.dropna(inplace=True)
    df["date"]=pd.to_datetime(df["date"],format="%d-%m-%Y")
    return df
def assign_company(df:pd.DataFrame)->pd.DataFrame:
    company_keywords={
        "RELIANCE": ["reliance"],
        "SBIN": ["sbi", "state bank"],
        "HDFCBANK": ["hdfc bank"],
        "INFY": ["infosys"],
        "TCS": ["tcs", "tata consultancy"],
        "ICICIBANK": ["icici bank"],
        "BHARTIARTL": ["airtel", "bharti"],
        "ADANIENT": ["adani"]
    }
    df["company"]=None
    text=df["headline"].str.lower()
    for company,keywords in company_keywords.items():
        mask=text.apply(lambda x:any(k in x for k in keywords))
        df.loc[mask,"company"]=company
    df=df[df["company"].notna()]
    return df
def map_to_trading_day(date:pd.Timestamp)->pd.Timestamp:
    if date.weekday()==5:
        return date+timedelta(days=2)
    elif date.weekday()==6:
        return date+timedelta(days=1)
    return date
def apply_trading_day_mapping(df:pd.DataFrame)->pd.DataFrame:
    df["trading_day"]=df["date"].apply(map_to_trading_day)
    return df
def load_finbert():
    tokenizer=AutoTokenizer.from_pretrained(
        "ProsusAI/finbert"
    )
    model=AutoModel.from_pretrained(
        "ProsusAI/finbert"
    )
    model.eval()
    return tokenizer,model
def encode_headlines(headlines,tokenizer,model):
    inputs=tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs=model(**inputs)
    embeddings=outputs.last_hidden_state[:,0,:].numpy()
    return embeddings
def aggregate_daily_news(df:pd.DataFrame)->pd.DataFrame:
    tokenizer,model=load_finbert()
    records=[]
    grouped=df.groupby(["company","trading_day"])
    for (company,day),group in grouped:
        headlines=group["headline"].tolist()
        embeddings=encode_headlines(headlines,tokenizer,model)
        daily_embedding=embeddings.mean(axis=0)
        records.append({
            "company":company,
            "date":day,
            "news_embedding":daily_embedding,
            "news_volume":len(headlines),
            "has_news":1
        })
    return pd.DataFrame(records)
def save_daily_news(df:pd.DataFrame):
    path="data/processed/news/daily_news_embeddings.parquet"
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(path)
