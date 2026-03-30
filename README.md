🚀 Event-Driven Multimodal Stock Movement Prediction:
A production-ready machine learning system that predicts next-day stock price direction by combining financial news (FinBERT embeddings) and time-series market data (LSTM), with uncertainty estimation and backtesting.


📌 Problem Statement:
Financial markets are influenced by both:
>Quantitative signals (price, volume, indicators)
>Qualitative signals (news, events, sentiment)
This project builds a multimodal deep learning system that integrates both sources to predict:

🧠 Model Architecture:
News Headlines → FinBERT → Embeddings (768D)
                                     ↓
Market Data (OHLCV + indicators) → LSTM
                                     ↓
         Multimodal Fusion Layer
                     ↓
          Fully Connected Layers
                     ↓
       Binary Prediction (UP / DOWN)
                     ↓
     Monte Carlo Dropout → Confidence Score

📊 Features Used:
1.📈 Market Data:
a.OHLCV (Open, High, Low, Close, Volume)
b.Log returns
c.RSI
d.MACD
e.Moving averages (SMA 20, SMA 50)
f.Rolling volatility
2.📰 News Data:
a.Company-specific financial headlines
b.FinBERT embeddings
c.Daily aggregation of headlines
d.News volume
e.Has-news indicator

🏗️ Pipeline Overview:
Raw Data
   ↓
Preprocessing (news + market)
   ↓
Feature Engineering
   ↓
Multimodal Dataset Creation
   ↓
Model Training (PyTorch)
   ↓
Evaluation + Backtesting
   ↓
Real-time Inference Pipeline
   ↓
Streamlit Dashboard

⚙️ Key Components:
1.✅ Multimodal Learning Combines:
a.Time-series modeling (LSTM)
b.NLP embeddings (FinBERT)

✅ Uncertainty Estimation
Uses Monte Carlo Dropout to compute prediction + confidence - score.

✅ Real-Time Inference:
a.Live news ingestion (Google News RSS)
b.Real-time feature generation
c.On-demand predictions

✅ Deployment-Ready Architecture:
a.Streamlit UI
b.FastAPI backend
c.Docker containerization

📊 Results:
>Metric	Value:
1.Accuracy	~51%
2.Precision	~52%
3.Recall	~69%
4.F1 Score	~0.60
5.Win Rate (Backtesting)	~52%

🖥️ Demo Features:
1.Select a company
2.View latest financial news
3.Get prediction (UP/DOWN)
4.View confidence score
5.See trading recommendation

🧰 Tech Stack:
1.Python
2.PyTorch
3.Transformers (FinBERT)
4.yfinance
Streamlit
FastAPI
Docker

⚠️ Limitations:
1.News timestamps assumed to be pre-market close.
2.Limited coverage of Indian financial news APIs.
3.Market efficiency limits prediction accuracy.

How to run the project:
1. clone the repository in your own system.
2. Then open the terminal and run streamlit run app.py .
3. Go to localhost and you will be able to use the project.

📌 Disclaimer:
This project is for educational purposes only, It does not constitute financial advice.
