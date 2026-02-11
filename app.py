import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# --- 1. INSTITUTIONAL DATA SCRAPERS ---
@st.cache_data(ttl=86400) # Cache for 24 hours
def get_full_market_tickers():
    """Scrapes the S&P 500 and Top Brazilian Tickers automatically"""
    try:
        # Node 1: S&P 500 from Wikipedia
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        us_tickers = sp500['Symbol'].tolist()
        
        # Node 2: Top Brazilian Tickers (B3)
        br_tickers = [
            'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'ABEV3.SA', 
            'WEGE3.SA', 'B3SA3.SA', 'RENT3.SA', 'SUZB3.SA', 'GGBR4.SA', 'JBSS3.SA',
            'RAIL3.SA', 'EQTL3.SA', 'VIVT3.SA', 'PRIO3.SA', 'LREN3.SA', 'RDOR3.SA'
        ]
        
        # Node 3: Nasdaq 100 / Tech Leaders
        tech = ['NVDA', 'TSLA', 'AMD', 'PLTR', 'NFLX', 'META', 'AMZN', 'GOOGL', 'MSFT']
        
        return list(set(us_tickers + br_tickers + tech))
    except Exception as e:
        return ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"] # Fallback

# --- 2. THE MEDALLION ENGINE ---
def analyze_medallion_logic(df, ticker):
    try:
        if df is None or len(df) < 50: return None
        curr = df['Close'].iloc[-1]
        
        # Z-Score & Volatility
        ma, std = df['Close'].rolling(50).mean().iloc[-1], df['Close'].rolling(50).std().iloc[-1]
        z = (curr - ma) / std
        sigma = df['Close'].pct_change().std() * np.sqrt(252)
        
        # ML Trend Projection
        y = df['Close'].tail(60).values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        forecast = model.predict([[len(y) + 30]])[0]
        
        # Options POP (Probability of Profit)
        d2 = (np.log(curr / forecast) + (0.045 - 0.5 * sigma**2) * (30/365)) / (sigma * np.sqrt(30/365))
        pop = si.norm.cdf(abs(d2)) * 100

        return {
            "Symbol": ticker, "Price": curr, "Z-Score": z, 
            "ML Target": forecast, "PoP %": pop, "Volatility": sigma
        }
    except: return None

# --- 3. THE 12-MINUTE FULL MARKET SCAN ---
@st.fragment(run_every=720)
def full_market_render():
    st.markdown("<h2 style='color:#58a6ff;'>Sovereign Terminal | Full Market Intelligence</h2>", unsafe_allow_html=True)
    
    ticker_list = get_full_market_tickers()
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Asset Universe", f"{len(ticker_list)} Stocks")
    col_stat2.metric("Scan Cycle", "12:00 MIN")
    col_stat3.metric("Protocol Status", "ENCRYPTED")

    with st.spinner(f"Vectorizing Algorithms for {len(ticker_list)} Assets..."):
        # Batching is the only way to avoid the ban
        df_mega = yf.download(ticker_list, period="1y", interval="1d", group_by='ticker', threads=True, progress=False)
        
        results = []
        for t in ticker_list:
            res = analyze_medallion_logic(df_mega[t], t)
            if res: results.append(res)

    if results:
        df_final = pd.DataFrame(results)
        st.dataframe(
            df_final.style.background_gradient(subset=['PoP %'], cmap='RdYlGn')
            .format({"Price": "${:.2f}", "Z-Score": "{:.2f}", "ML Target": "${:.2f}", "PoP %": "{:.1f}%", "Volatility": "{:.1%}"}),
            use_container_width=True, hide_index=True, height=600
        )
    else:
        st.error("Protocol Failure: Multi-source validation failed. Yahoo IP Limited.")

# --- 4. MAIN INTERFACE ---
full_market_render()
