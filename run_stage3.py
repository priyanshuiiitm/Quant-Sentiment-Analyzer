from src.news_pipeline import(
    load_raw_news,
    assign_company,
    apply_trading_day_mapping,
    aggregate_daily_news,
    save_daily_news,
)
Raw_news_path="Data/raw/news/economic_times.csv"
df=load_raw_news(Raw_news_path)
df=assign_company(df)
df=apply_trading_day_mapping(df)
daily_news=aggregate_daily_news(df)
save_daily_news(daily_news)
print("stage 3 completed")