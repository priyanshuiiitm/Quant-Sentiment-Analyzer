import torch
from torch.utils.data import Dataset
import numpy as np
class StockDataset(Dataset):
    def __init__(self,df):
        self.market=df["market_window"].values
        self.news=df["news_embedding"].values
        self.news_volume=df["news_volume"].values
        self.has_news=df["has_news"].values
        self.labels=df["label"].values
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        market=torch.tensor(np.stack(self.market[idx]).astype(np.float32),dtype=torch.float32)
        news=torch.tensor(np.array(self.news[idx],dtype=np.float32),dtype=torch.float32)
        news_volume=torch.tensor([self.news_volume[idx]],dtype=torch.float32)
        has_news=torch.tensor([self.has_news[idx]],dtype=torch.float32)
        label=torch.tensor(self.labels[idx],dtype=torch.float32)
        return market,news,news_volume,has_news,label
