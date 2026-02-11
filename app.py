import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="Fidelity Mega-Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1217; color: #e1e4e8; }
    .stMetric { background-color: #161c23; border: 1px solid #30363d; padding: 15px; border-radius: 4px; }
    .stDataFrame { border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MULTI-SOURCE DATA ENGINE ---

def get_mega_watchlist():
    # US Markets (S&P 500 leaders)
    us = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "PLTR", "COIN", "BRK-B", "V", "JPM", "WMT", "COST", "DIS", "PYPL", "BA", "INTC", "CSCO", "PEP", "KO", "XOM", "CVX"]
    # Brazil Markets
    br = ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA", "WEGE3.SA", "PRIO3.SA", "RENT3.SA", "GGBR4.SA", "JBSS3.SA", "ELET3.SA", "CSAN3.SA", "SUZB3.SA"]
    # Adding more to hit the "hundreds" mark
    return list(set(us + br + ["SPY", "QQQ", "IWM", "EWZ", "GLD", "SLV", "BTC-USD"]))

def fetch_finviz_news(ticker):
    """Spreads the load by fetching news from FinViz instead of Yahoo"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.split('.')[0]}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.content, 'html.parser')
        news_table = soup.find(id='news-table')
        return [row.a.text for row in news_table.findAll('tr')[:3]]
    except: return ["No recent news found on FinViz."]

# --- 3. ALGORITHMIC CALCULATIONS ---

def analyze_assets(df_mega, ticker_list):
    results = []
    for ticker in ticker_list:
        try:
            df = df_mega[ticker].dropna()
            if df.empty or len(df) < 30: continue
            
            # --- VWAP ---
            df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # --- TOC SQUEEZE (Bollinger vs Keltner) ---
            sma = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            atr = (df['High'] - df['Low']).rolling(20).mean()
            squeeze = "🔒 SQUEEZE" if (sma - (2*std) > sma - (1.5*atr)).iloc[-1] else "🌊 EXPANSION"
            
            # --- ML PRICE FORECAST ---
            y = df['Close'].values
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            ml_target = model.predict([[len(y) + 1]])[0]
            
            # --- SUCCESS PROBABILITY ---
            sigma = df['Close'].pct_change().std() * np.sqrt(252 * 7)
            dist = np.log(ml_target / df['Close'].iloc[-1])
            pop = si.norm.cdf(dist / (sigma * np.sqrt(1/12))) * 100

            results.append({
                "Ticker": ticker, "Price": df['Close'].iloc[-1], "VWAP": df['vwap'].iloc[-1],
                "TOC State": squeeze, "ML Target": ml_target, "Success %": pop
            })
        except: continue
    return pd.DataFrame(results)

# --- 4. TERMINAL FRAGMENT (Auto-Refresh every 60s) ---

@st.fragment(run_every=60)
def live_terminal():
    st.title("💹 Fidelity Mega-Terminal | Live Intelligence")
    st.write(f"🕒 **Last Terminal Sync:** {datetime.now().strftime('%H:%M:%S')}")
    
    ticker_universe = get_mega_watchlist()
    
    with st.spinner(f"Vectorizing Algos for {len(ticker_universe)} assets..."):
        # MEGA-DOWNLOAD (Batching prevents the ban)
        df_mega = yf.download(ticker_universe, period="60d", interval="1h", group_by='ticker', threads=True, progress=False)
        
        if not df_mega.empty:
            df_final = analyze_assets(df_mega, ticker_universe)
            
            tab_scanner, tab_news, tab_math = st.tabs(["🚀 ALGO SCANNER", "📰 RESEARCH TERMINAL", "🧮 MATH CLASSROOM"])
            
            with tab_scanner:
                st.subheader("Elite Market Scanner")
                st.dataframe(
                    df_final.style.background_gradient(subset=['Success %'], cmap='RdYlGn')
                    .format({"Price": "{:.2f}", "VWAP": "{:.2f}", "ML Target": "{:.2f}", "Success %": "{:.1f}%"}),
                    use_container_width=True, hide_index=True, height=600
                )

            with tab_news:
                st.subheader("Global Intelligence News Feed (FinViz/NASDAQ)")
                selected_ticker = st.selectbox("Select Asset for Deep Research", ticker_universe)
                news = fetch_finviz_news(selected_ticker)
                for item in news:
                    st.write(f"• {item}")

            with tab_math:
                st.markdown("### Terminal Logic Matrix")
                st.write("**TOC Squeeze:** Detects volatility compression.")
                st.write("**VWAP:** Institutional benchmark for fair pricing.")
                st.write("**ML Target:** 1-hour linear projection.")

live_terminal()
