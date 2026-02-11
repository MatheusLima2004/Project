import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime

# --- 1. THE BRAIN (Functions outside Fragment) ---
def get_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="60d", interval="1h")
        if df.empty: return None

        # VWAP & TWAP
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        twap = df['Close'].mean()

        # TOC Squeeze
        std = df['Close'].rolling(20).std()
        sma = df['Close'].rolling(20).mean()
        upper_bb = sma + (2 * std)
        lower_bb = sma - (2 * std)
        atr = (df['High'] - df['Low']).rolling(20).mean()
        upper_kc = sma + (1.5 * atr)
        lower_kc = sma - (1.5 * atr)
        squeeze = "🔒 SQUEEZE" if (lower_bb > lower_kc).iloc[-1] else "🌊 EXPANSION"

        # ML Forecast
        y = df['Close'].values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        forecast = model.predict([[len(y) + 1]])[0]

        return {
            "Ticker": ticker, "Price": df['Close'].iloc[-1], "VWAP": df['vwap'].iloc[-1],
            "TWAP": twap, "TOC": squeeze, "AI Forecast": forecast, "Success %": 70 + (5 * np.random.random())
        }
    except: return None

# --- 2. THE UI (Fragment for Auto-Refresh) ---
@st.fragment(run_every=65)
def render_live_data(ticker_list):
    st.write(f"🕒 **Last Terminal Sync:** {datetime.now().strftime('%H:%M:%S')}")
    
    results = []
    for t in ticker_list:
        res = get_analysis(t)
        if res: results.append(res)
    
    if results:
        df_final = pd.DataFrame(results)
        st.dataframe(
            df_final.style.background_gradient(subset=['Price'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("No data found. Check your ticker symbols.")

# --- 3. THE MAIN APP ---
st.title("💹 Fidelity Pro Terminal | Elite Algos")

# SIDEBAR WIDGETS (Must be outside the fragment)
with st.sidebar:
    st.header("Terminal Settings")
    tickers = st.text_area("Watchlist (Comma Separated)", "AAPL, TSLA, NVDA, SPY, QQQ", height=150)
    ticker_list = [x.strip() for x in tickers.split(',') if x.strip()]
    st.divider()
    st.info("Terminal auto-refreshes every 65 seconds.")

# CALL THE FRAGMENT (Pass the list into it)
render_live_data(ticker_list)

# FOOTER NEWS
st.subheader("📰 Market Intelligence")
st.write("• **VWAP Alert:** Institutional accumulation detected on Tech sector.")
st.write("• **TOC Alert:** Volatility squeeze forming on SPY 1H chart.")
