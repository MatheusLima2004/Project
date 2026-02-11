import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# --- 1. COMMAND CENTER STYLING ---
st.set_page_config(page_title="Sovereign Elite Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #e1e4e8; }
    [data-testid="stSidebar"] { background-color: #15191e; border-right: 1px solid #2d333b; }
    .stMetric { background-color: #15191e; border: 1px solid #2d333b; padding: 15px; border-radius: 4px; }
    .intel-card { background-color: #15191e; padding: 12px; border-radius: 4px; border-left: 4px solid #00c805; margin-bottom: 8px; font-size: 13px; }
    .algo-label { color: #8b949e; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DEEP-SCAN ALGORITHM ENGINE ---
def analyze_sovereign_logic(df, ticker):
    try:
        if df.empty or len(df) < 50: return None
        curr = df['Close'].iloc[-1]
        
        # 1. Z-Score (Mean Reversion)
        ma_50 = df['Close'].rolling(50).mean()
        std_50 = df['Close'].rolling(50).std()
        z = (curr - ma_50.iloc[-1]) / std_50.iloc[-1]
        
        # 2. Kelly Criterion (Sizing)
        rets = df['Close'].pct_change().dropna()
        win_rate = len(rets[rets > 0]) / len(rets)
        win_loss = rets[rets > 0].mean() / abs(rets[rets < 0].mean()) if not rets[rets < 0].empty else 1
        kelly = (win_rate * (win_loss + 1) - 1) / win_loss
        
        # 3. ML Trend & PoP (Options Intelligence)
        y = df['Close'].tail(60).values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        target = model.predict([[len(y) + 30]])[0]
        sigma = rets.std() * np.sqrt(252)
        d2 = (np.log(curr / target) + (0.045 - 0.5 * sigma**2) * (30/365)) / (sigma * np.sqrt(30/365))
        pop = si.norm.cdf(abs(d2)) * 100

        return {
            "Symbol": ticker, "Price": curr, "Z-Score": z, 
            "Kelly %": max(0, kelly * 100), "AI Target": target, "PoP %": pop
        }
    except: return None

# --- 3. 12-MINUTE SOVEREIGN FRAGMENT ---
@st.fragment(run_every=720)
def render_command_center(ticker_list):
    st.markdown("<h2 style='color:#58a6ff;'>Sovereign Command Center | Institutional Intelligence</h2>", unsafe_allow_html=True)
    
    # Global Pulse Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Node Status", "DECENTRALIZED", delta="ACTIVE")
    m2.metric("Scan Cycle", "12:00 MIN", delta="DEEP ML")
    m3.metric("Assets Online", len(ticker_list))
    m4.metric("Sync Time", datetime.now().strftime("%H:%M:%S"))

    with st.spinner("Mining Multi-Node Data Mines..."):
        # Batching for full-market throughput
        df_mega = yf.download(ticker_list, period="2y", interval="1d", group_by='ticker', threads=True, progress=False)
        
        results, intel_feed = [], []
        for t in ticker_list:
            res = analyze_sovereign_logic(df_mega[t], t)
            if res: 
                results.append(res)
                try:
                    # Offload intelligence to news node
                    n = yf.Ticker(t).news[:1]
                    for item in n: intel_feed.append({"s": t, "t": item['title'], "l": item['link']})
                except: pass

    # Platform Grid
    col_main, col_intel = st.columns([3, 1])
    
    with col_main:
        st.markdown("<p class='algo-label'>Global Intelligence Matrix</p>", unsafe_allow_html=True)
        df_res = pd.DataFrame(results)
        
        # Fail-safe styling check
        styled_df = df_res.style
        if 'PoP %' in df_res.columns:
            styled_df = styled_df.background_gradient(subset=['PoP %'], cmap='RdYlGn')
        
        st.dataframe(
            styled_df.format({"Price": "${:.2f}", "Z-Score": "{:.2f}", "Kelly %": "{:.1f}%", "AI Target": "${:.2f}", "PoP %": "{:.1f}%"}),
            use_container_width=True, hide_index=True, height=500
        )
        
        # Live Stream Module
        st.markdown("<p class='algo-label'>Execution Stream</p>", unsafe_allow_html=True)
        st.components.v1.html(f"""
            <div id="tv-chart" style="height:400px;"></div>
            <script src="https://s3.tradingview.com/tv.js"></script>
            <script>
            new TradingView.widget({{"width": "100%", "height": 400, "symbol": "{ticker_list[0]}", "interval": "D", "theme": "dark", "style": "1", "container_id": "tv-chart"}});
            </script>
        """, height=400)

    with col_intel:
        st.markdown("<p class='algo-label'>Institutional Intel Feed</p>", unsafe_allow_html=True)
        for news in intel_feed[:15]:
            st.markdown(f"""<div class='intel-card'><b>{news['s']}</b>: <a href='{news['l']}' target='_blank' style='color:white; text-decoration:none;'>{news['t']}</a></div>""", unsafe_allow_html=True)

# --- 4. ASSET UNIVERSE INPUT ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e3/Fidelity_Investments_logo.svg", width=120)
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Asset Universe")
    default_list = "AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, NFLX, AMD, PLTR, VALE3.SA, PETR4.SA, ITUB4.SA, SPY, QQQ"
    ticker_list = [x.strip() for x in st.text_area("Ticker List", default_list, height=300).split(',') if x.strip()]

render_command_center(ticker_list)
