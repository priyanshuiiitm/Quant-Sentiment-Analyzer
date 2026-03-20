import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
from src.dataset.stock_dataset import StockDataset
from src.models.multimodal_model import MultimodalStockModel
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
def train_model():
    train_df=pd.read_parquet("Data/processed/final/train.parquet")
    val_df=pd.read_parquet("Data/processed/final/val.parquet")
    train_dataset=StockDataset(train_df)
    val_dataset=StockDataset(val_df)
    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=MultimodalStockModel().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    criterion=nn.BCELoss()
    epochs=50
    for epoch in range(epochs):
        model.train()
        total_loss=0
        for market,news,vol,has_news,label in train_loader:
            market=market.to(device)
            news=news.to(device)
            vol=vol.to(device)
            has_news=has_news.to(device)
            label=label.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred=model(market,news,vol,has_news)
            loss=criterion(pred,label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total_loss+=loss.item()
        print(f"Epoch {epoch+1} train loss {total_loss/len(train_loader)}")
        evaluate(model,val_loader,device)
    torch.save(model.state_dict(),"model.pth")
    print("Model saved to model.pth")
def evaluate(model,loader,device):
    model.eval()
    correct=0
    total=0
    all_preds=[]
    all_labels=[]
    with torch.no_grad():
        for market,news,vol,has_news,label in loader:
            market=market.to(device)
            news=news.to(device)
            vol=vol.to(device)
            has_news=has_news.to(device)
            preds=model(market,news,vol,has_news)
            preds=(preds>0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(label.numpy().flatten())
    accuracy=accuracy_score(all_labels,all_preds)
    precision=precision_score(all_labels,all_preds,zero_division=0)
    recall=recall_score(all_labels,all_preds,zero_division=0)
    f1=f1_score(all_labels,all_preds,zero_division=0)
    print(f"Validation accuracy:{accuracy:.4f}")
    print(f"Precision:{precision:.4f}")
    print(f"Recall:{recall:.4f}")
    print(f"F1 score:{f1:.4f}")
def mc_dropout_predict(model,market,news,vol,has_news,passes=20):
    model.train()
    preds=[]
    for k in range(passes):
        with torch.no_grad():
            pred=model(market,news,vol,has_news)
            preds.append(pred.cpu().numpy())
    preds=np.array(preds)
    mean_pred=preds.mean()
    uncertainty=preds.std()
    confidence=1-uncertainty
    return mean_pred,confidence