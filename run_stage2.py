from src.data_pipeline import(
    fetch_ohlcv,
    add_technical_indicators,
    save_processed_market_data,
)
from src.companies import COMPANIES
START_DATE="2022-01-01"
END_DATE="2025-01-01"
for company,ticker in COMPANIES.items():
    print(f"Processing {company}...")
    raw_path=f"Data/raw/market/{company}.csv"
    df=fetch_ohlcv(
        ticker=ticker,
        start=START_DATE,
        end=END_DATE,
        save_path=raw_path,
    )
    if df is None or df.empty or "Close" not in df.columns:
        print(f"skipping {company}:no market data downloaded.")
        continue
    df=add_technical_indicators(df)
    save_processed_market_data(df,company)
print("stage 2 completed.")