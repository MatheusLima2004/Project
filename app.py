import streamlit as st
import pandas as pd
from datetime import datetime
import yfinance as yf
from engine import get_full_market_tickers, run_medallion_math

# --- 1. UI CONFIG ---
st.set_page_config(page_title="Sovereign Command Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #e1e4e8; }
    .stMetric { background-color: #15191e; border: 1px solid #2d333b; padding: 15px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DEEP SCAN FRAGMENT (12-MIN CYCLE) ---
@st.fragment(run_every=720)
def render_terminal():
    st.title("💹 Sovereign Elite | Command Center")
    
    ticker_list = get_full_market_tickers()
    
    # Platform Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Universe", f"{len(ticker_list)} Assets")
    c2.metric("Cycle Status", "12-MIN DEEP SCAN")
    c3.metric("Last Sync", datetime.now().strftime("%H:%M:%S"))

    with st.spinner("Synchronizing Multi-Node Data Mines..."):
        df_mega = yf.download(ticker_list, period="1y", interval="1d", group_by='ticker', threads=True, progress=False)
        
        results = []
        for t in ticker_list:
            res = run_medallion_math(df_mega[t], t)
            if res: results.append(res)
    
    if results:
        df_final = pd.DataFrame(results)
        st.subheader("📊 Global Intelligence Matrix")
        st.dataframe(
            df_final.style.background_gradient(subset=['PoP %'], cmap='RdYlGn')
            .format({"Price": "${:.2f}", "Z-Score": "{:.2f}", "Kelly %": "{:.1f}%", "ML Target": "${:.2f}", "PoP %": "{:.1f}%"}),
            use_container_width=True, hide_index=True, height=600
        )
        
        # Live Chart
        st.subheader(f"📈 Execution Feed: {ticker_list[0]}")
        st.components.v1.html(f"""
            <div id="tv-chart" style="height:400px;"></div>
            <script src="https://s3.tradingview.com/tv.js"></script>
            <script>
            new TradingView.widget({{"width": "100%", "height": 400, "symbol": "{ticker_list[0]}", "interval": "D", "theme": "dark", "style": "1", "container_id": "tv-chart"}});
            </script>
        """, height=400)

# --- 3. EXECUTION ---
render_terminal()
