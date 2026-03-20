import pandas as pd
def run_backtest(df):
    capital=10000
    capital_history=[capital]
    trades=0
    wins=0
    df=df.sort_values(["company","date"]).reset_index(drop=True)
    for company,group in df.groupby("company"):
        group=group.reset_index(drop=True)
        for i in range(len(df)-1):
            pred=df.iloc[i]["prediction"]
            close_today=df.iloc[i]["close_price"]
            close_next=df.iloc[i+1]["close_price"]
            if pred>0.5:
                trades+=1
                return_pct=(close_next-close_today)/close_today
                capital=capital*(1+return_pct)
                if return_pct>0:
                    wins+=1
            capital_history.append(capital)
    win_rate=wins/trades if trades >0 else 0
    total_return=(capital-10000)/10000
    print("Total return:",total_return)
    print("Trades:",trades)
    print("Win Rate:",win_rate)
    return capital_history