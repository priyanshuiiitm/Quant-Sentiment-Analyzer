import torch 
import pandas as pd
from src.dataset.stock_dataset import StockDataset
from src.models.multimodal_model import MultimodalStockModel
def load_model(path,device):
    model=MultimodalStockModel().to(device)
    model.load_state_dict(torch.load(path,map_location=device))
    model.eval()
    return model
def predict_dataset(model,df,device):
    dataset=StockDataset(df)
    preds=[]
    for i in range(len(dataset)):
        market,news,vol,has_news,_=dataset[i]
        market=market.unsqueeze(0).to(device)
        news=news.unsqueeze(0).to(device)
        vol=vol.unsqueeze(0).to(device)
        has_news=has_news.unsqueeze(0).to(device)
        with torch.no_grad():
            pred=model(market,news,vol,has_news)
        preds.append(pred.item())
    df["prediction"]=preds
    return df