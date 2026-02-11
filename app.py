import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# --- 1. SOVEREIGN THEME ---
st.set_page_config(page_title="Medallion Sovereign Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #e1e4e8; }
    .stMetric { background-color: #15191e; border: 1px solid #2d333b; padding: 15px; border-radius: 4px; }
    .strategy-card { background-color: #1c2128; border-left: 5px solid #58a6ff; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DEEP OPTIONS & ML LOGIC ---
def analyze_options_strategy(df, ticker):
    try:
        if df is None or len(df) < 50: return None
        curr = df['Close'].iloc[-1]
        
        # Z-Score (Mean Reversion)
        ma_50 = df['Close'].rolling(50).mean()
        std_50 = df['Close'].rolling(50).std()
        z = (curr - ma_50.iloc[-1]) / std_50.iloc[-1]
        
        # ML Trend (30-Day Projection)
        y = df['Close'].tail(60).values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        forecast = model.predict([[len(y) + 30]])[0]
        
        # Options ML: PoP (Probability of Profit)
        rets = df['Close'].pct_change().dropna()
        sigma = rets.std() * np.sqrt(252)
        days_to_expiry = 30 / 365
        d2 = (np.log(curr / forecast) + (0.045 - 0.5 * sigma**2) * days_to_expiry) / (sigma * np.sqrt(days_to_expiry))
        pop = si.norm.cdf(abs(d2)) * 100
        
        # TOC Squeeze (TTM Logic)
        atr = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
        squeeze = "🔒 SQUEEZE" if (2 * std_50.iloc[-1] < 1.5 * atr) else "🌊 EXPANSION"

        # Strategy Logic
        if z < -1.5 and pop > 60: strat = "BULLISH CALL"
        elif z > 1.5 and pop < 40: strat = "BEARISH PUT"
        else: strat = "NEUTRAL / HOLD"

        return {
            "Symbol": ticker, "Price": curr, "Z-Score": z, 
            "Kelly %": 0.0, "State": squeeze, "ML Target": forecast, 
            "PoP %": pop, "Strategy": strat
        }
    except: return None

# --- 3. THE FAIL-SAFE RENDER ENGINE ---
@st.fragment(run_every=900)
def sovereign_render(ticker_list):
    st.markdown("<h2 style='color:#58a6ff;'>Sovereign Command | Deep ML Options Matrix</h2>", unsafe_allow_html=True)
    
    with st.spinner("Executing Staggered Multi-Node Intelligence..."):
        # Download data (Fail-safe Batching)
        df_mega = yf.download(ticker_list, period="2y", interval="1d", group_by='ticker', threads=True, progress=False)
        
        results = []
        for t in ticker_list:
            res = analyze_options_strategy(df_mega[t], t)
            if res: results.append(res)
        
    if results:
        df_res = pd.DataFrame(results)
        
        # --- THE FIX: DYNAMIC STYLING ---
        styled_df = df_res.style
        
        # Only style if columns exist (Prevents KeyError)
        if 'PoP %' in df_res.columns:
            styled_df = styled_df.background_gradient(subset=['PoP %'], cmap='RdYlGn')
        
        if 'Strategy' in df_res.columns:
            styled_df = styled_df.map(
                lambda x: 'color: #00c805' if x == 'BULLISH CALL' else ('color: #ff3b3b' if x == 'BEARISH PUT' else ''),
                subset=['Strategy']
            )

        st.dataframe(
            styled_df.format({
                "Price": "${:.2f}", "Z-Score": "{:.2f}", "ML Target": "${:.2f}", "PoP %": "{:.1f}%"
            }),
            use_container_width=True, hide_index=True, height=500
        )
    else:
        st.error("Protocol Error: No data processed. Check Ticker List in Sidebar.")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Asset Universe")
    default_list = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, AMD, PLTR, VALE3.SA, PETR4.SA, ITUB4.SA, SPY, QQQ"
    raw_tickers = st.text_area("Ticker List", default_list, height=300)
    ticker_list = [x.strip() for x in raw_tickers.split(',') if x.strip()]

sovereign_render(ticker_list)
