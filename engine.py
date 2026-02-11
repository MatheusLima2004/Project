import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression

def get_full_market_tickers():
    """Scrapes S&P 500 and B3 automatically"""
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        us = sp500['Symbol'].tolist()
        br = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'WEGE3.SA']
        return list(set(us + br))
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]

def run_medallion_math(df, ticker):
    """The Proprietary Algorithm Matrix"""
    try:
        if df is None or len(df) < 50: return None
        curr = df['Close'].iloc[-1]
        
        # Z-Score (Mean Reversion)
        ma, std = df['Close'].rolling(50).mean().iloc[-1], df['Close'].rolling(50).std().iloc[-1]
        z = (curr - ma) / std
        
        # Kelly Criterion (Position Sizing)
        rets = df['Close'].pct_change().dropna()
        win_rate = len(rets[rets > 0]) / len(rets)
        win_loss = rets[rets > 0].mean() / abs(rets[rets < 0].mean()) if not rets[rets < 0].empty else 1
        kelly = (win_rate * (win_loss + 1) - 1) / win_loss
        
        # ML Trend & Options PoP (30-Day Outlook)
        y = df['Close'].tail(60).values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        target = model.predict([[len(y) + 30]])[0]
        sigma = rets.std() * np.sqrt(252)
        d2 = (np.log(curr / target) + (0.045 - 0.5 * sigma**2) * (30/365)) / (sigma * np.sqrt(30/365))
        pop = si.norm.cdf(abs(d2)) * 100

        return {
            "Symbol": ticker, "Price": curr, "Z-Score": z, 
            "Kelly %": max(0, kelly * 100), "ML Target": target, "PoP %": pop
        }
    except: return None
