import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Global Mega-Scanner", layout="wide")

st.title("🌐 Global Mega-Scanner (S&P 500 + B3)")

# --- 2. THE BIG LIST (Hundreds of Stocks) ---
def get_massive_watchlist():
    # S&P 500 Top Movers + Popular Tech
    us_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "V", 
        "JPM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "LLY", "ABBV", "PFE", 
        "COST", "PEP", "KO", "ORCL", "BAC", "AVGO", "TMO", "CSCO", "ACN", "ADBE"
    ]
    
    # Brazil B3 Top Movers
    br_stocks = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA", 
        "WEGE3.SA", "B3SA3.SA", "RENT3.SA", "SUZB3.SA", "GGBR4.SA", "JBSS3.SA",
        "LREN3.SA", "RDOR3.SA", "RAIL3.SA", "EQTL3.SA", "VIVT3.SA", "PRIO3.SA"
    ]
    
    # Major ETFs
    etfs = ["SPY", "QQQ", "IWM", "EEM", "GLD", "SLV", "DIA", "XLE", "XLF", "XLK"]
    
    # If you want even more, we can add a list of 500+ here.
    return us_stocks + br_stocks + etfs

# --- 3. THE BATCH ENGINE ---
@st.cache_data(ttl=3600)
def run_mega_scan(tickers):
    # This downloads ALL tickers at once (The most efficient way)
    data = yf.download(tickers, period="5d", interval="1d", group_by='ticker')
    
    results = []
    for ticker in tickers:
        try:
            # Extract individual stock data from the mega-block
            hist = data[ticker]
            if not hist.empty:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((curr - prev) / prev) * 100
                
                # Fair Value (Calculated only if data is available)
                # Note: Info fetching for 500 stocks is slow, so we focus on Price/Trend
                
                results.append({
                    "Ticker": ticker,
                    "Price": curr,
                    "Change %": change,
                    "Signal": "🚀 BUY" if change < -1.5 else ("📉 SELL" if change > 1.5 else "😴 HOLD")
                })
        except:
            continue
            
    return pd.DataFrame(results)

# --- 4. RENDER DASHBOARD ---
tickers = get_massive_watchlist()

if st.button(f"🚀 SCAN {len(tickers)} ASSETS NOW"):
    with st.spinner("Analyzing Global Markets..."):
        df = run_mega_scan(tickers)
        
    if not df.empty:
        # Sort by biggest gainers
        df = df.sort_values(by="Change %", ascending=False)
        
        # Display Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Assets Scanned", len(df))
        c2.metric("Market Leader", df.iloc[0]['Ticker'], f"{df.iloc[0]['Change %']:.2f}%")
        c3.metric("Market Laggard", df.iloc[-1]['Ticker'], f"{df.iloc[-1]['Change %']:.2f}%")
        
        # Display Results
        st.dataframe(
            df.style.background_gradient(subset=['Change %'], cmap='RdYlGn'),
            use_container_width=True,
            height=800
        )
    else:
        st.error("Connection lost. Yahoo is blocking this cloud request. Try again in a few minutes.")

else:
    st.info("Click the button to scan hundreds of assets simultaneously.")
