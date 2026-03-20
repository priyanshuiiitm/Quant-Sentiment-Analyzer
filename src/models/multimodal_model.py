import torch
import torch.nn as nn
class MultimodalStockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=12,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.news_fc=nn.Sequential(
            nn.Linear(768,64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier=nn.Sequential(
            nn.Linear(64+128+2,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,1)
        )
    def forward(self,market,news,news_volume,has_news):
        lstm_out,_=self.lstm(market)
        market_feat=lstm_out[:,-1,:]
        news_feat=self.news_fc(news)
        combined=torch.cat(
            [market_feat,news_feat,news_volume,has_news],
            dim=1
        )
        output=self.classifier(combined)
        return torch.sigmoid(output)

    