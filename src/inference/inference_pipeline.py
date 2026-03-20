import torch
import numpy as np
from src.inference.news_fetcher import fetch_company_news
from src.inference.embedding_service import encode_headlines
from src.inference.market_service import get_market_features
from src.models.multimodal_model import MultimodalStockModel
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=MultimodalStockModel().to(device)
model.load_state_dict(torch.load("model.pth",map_location=device))
model.eval()
def predict_stock(company,ticker):
    headlines=fetch_company_news(company)
    print("DEBUG company:", company)                                                                                                                             
    print("DEBUG headlines count:", len(headlines))                                                                                                              
    print("DEBUG headlines:", headlines)                                                                                                                         
    news_embedding=encode_headlines(headlines)
    market_window=get_market_features(ticker)
    news_volume=len(headlines)
    has_news=1 if news_volume >0 else 0
    market_tensor=torch.tensor(market_window,dtype=torch.float32).unsqueeze(0).to(device)
    news_tensor=torch.tensor(news_embedding,dtype=torch.float32).unsqueeze(0).to(device)
    volume_tensor=torch.tensor([[news_volume]],dtype=torch.float32).to(device)
    news_flag=torch.tensor([[has_news]],dtype=torch.float32).to(device)
    probability,confidence=mc_dropout_prediction(model,market_tensor,news_tensor,volume_tensor,news_flag)
    direction="UP" if probability>0.5 else "DOWN"
    return{
        "company":company,
        "prediction":direction,
        "probability":probability,
        "confidence":confidence,
        "headlines":headlines
    }
def mc_dropout_prediction(model,market_tensor,news_tensor,volume_tensor,news_flag,passes=20):
    model.train()
    predictions=[]
    for _ in range(passes):
        with torch.no_grad():
            pred=model(market_tensor,news_tensor,volume_tensor,news_flag)
        predictions.append(pred.item())
    predictions=np.array(predictions)
    mean_prediction=predictions.mean()
    uncertainity=predictions.std()
    confidence=1/(1+uncertainity)
    return mean_prediction,confidence


