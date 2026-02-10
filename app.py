import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import math

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Max Capacity Terminal", page_icon="🌎", layout="wide")
st.title("🌎 The Global Value Scanner (Max Capacity)")
st.markdown("### 🔍 Scanning S&P 500 + NASDAQ + Brazil | 📊 Fair Value Analysis")

# --- 2. DATA FETCHING (THE MAX LIST) ---
@st.cache_data(ttl=3600)
def get_universe():
    # 1. Fetch S&P 500 & Nasdaq dynamically
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        sp500 = [x.replace('.', '-') for x in sp500] # Fix BRK.B
    except:
        sp500 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V"]

    # 2. Brazil Ibovespa (Top Liquid Manual List to ensure correctness)
    brazil = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "WEGE3.SA",
        "ABEV3.SA", "RENT3.SA", "BPAC11.SA", "SUZB3.SA", "HAPV3.SA", "RDOR3.SA",
        "B3SA3.SA", "EQTL3.SA", "LREN3.SA", "PRIO3.SA", "RAIL3.SA", "GGBR4.SA",
        "JBSS3.SA", "RADL3.SA", "VBBR3.SA", "CSAN3.SA", "TOTS3.SA", "BBSE3.SA",
        "EMBJ3.SA", "VIVT3.SA", "CMIG4.SA", "ELET3.SA", "SBSP3.SA", "TIMS3.SA",
        "MGLU3.SA", "PETZ3.SA", "AZUL4.SA", "GOLL4.SA", "CRFB3.SA", "NTCO3.SA",
        "EGIE3.SA", "TAEE11.SA", "CPLE6.SA", "CSNA3.SA", "USIM5.SA", "KLBN11.SA"
    ]
    
    # 3. ETFs
    etfs = ["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "XLE", "XLF", "SMH", "ARKK"]
    
    # Combine unique tickers
    return list(set(sp500 + brazil + etfs))

# --- 3. MATH FORMULAS ---
def calculate_fair_value(eps, book_value):
    """Benjamin Graham Number: Sqrt(22.5 * EPS * Book Value)"""
    if eps > 0 and book_value > 0:
        return math.sqrt(22.5 * eps * book_value)
    return 0

def black_scholes(S, K, T, r, sigma, option_type="call"):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return max(S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2), 0)
        else:
            return max(K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1), 0)
    except: return 0

def generate_sparkline(series):
    bar_chars = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    if series.empty: return ""
    series = series.tail(30)
    min_val, max_val = series.min(), series.max()
    if max_val == min_val: return "▇" * 10
    spark = ""
    for price in series:
        idx = int((price - min_val) / (max_val - min_val) * 7)
        spark += bar_chars[idx]
    return spark

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=4.5) / 100
    st.info("⚠️ Scanning ~600+ stocks. This will take 2-3 minutes.")

# --- 5. ENGINE ---
@st.cache_data(ttl=3600)
def scan_market():
    tickers = get_universe()
    data_list = []
    
    progress_text = f"Scanning {len(tickers)} assets... Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    # Batch Download (Much Faster)
    try:
        hist_data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except: return pd.DataFrame()
    
    for i, ticker in enumerate(tickers):
        try:
            # 1. Get Price History
            df = hist_data[ticker] if len(tickers) > 1 else hist_data
            if df.empty or len(df) < 50: continue
            
            close = df['Close'].dropna()
            current_price = close.iloc[-1]
            daily_returns = close.pct_change().dropna()
            hist_vol = daily_returns.std() * np.sqrt(252)
            spark = generate_sparkline(close)
            
            # 2. Get Fundamentals (Slow Step)
            info = yf.Ticker(ticker).info
            eps = info.get('trailingEps', 0)
            bvps = info.get('bookValue', 0)
            
            # 3. Calculate FAIR VALUE
            fair_value = calculate_fair_value(eps, bvps)
            
            # Margin of Safety % (Positive = Good, Negative = Overpriced)
            if fair_value > 0:
                safety_margin = ((fair_value - current_price) / fair_value) * 100
            else:
                safety_margin = -999 # Undefined (Growth stock or loss maker)

            # 4. Classify Asset
            if ".SA" in ticker:
                country = "🇧🇷 Brazil"
                currency = "R$"
            elif hist_vol < 0.20 and "ETF" in info.get('quoteType', ''):
                country = "🛡️ ETF"
                currency = "$"
            else:
                country = "🇺🇸 USA"
                currency = "$"

            # 5. Options (Simplified for Speed)
            # Only check options if Margin of Safety is interesting (> 0%)
            best_opt = "N/A"
            if safety_margin > 0 and current_price > 5:
                # Placeholder for complex options logic (skipped to save time on 600 stocks)
                best_opt = "Check Chain"

            data_list.append({
                "Ticker": ticker,
                "Region": country,
                "Price": current_price,
                "Fair Value": fair_value,
                "Safety Margin %": safety_margin,
                "Trend": spark,
                "Volatility": hist_vol
            })
            
        except: continue
        
        # Update progress every 10 stocks to save UI rendering time
        if i % 10 == 0:
            my_bar.progress(min((i + 1) / len(tickers), 1.0))
            
    my_bar.empty()
    return pd.DataFrame(data_list)

# --- 6. RENDER ---
if st.button("🚀 Run Max Capacity Scan"):
    st.cache_data.clear()

df = scan_market()

if not df.empty:
    tab1, tab2, tab3 = st.tabs(["🇺🇸 S&P 500 / NASDAQ", "🇧🇷 Brazil (B3)", "🛡️ ETFs"])
    
    # CONFIGURATION FOR TABLES
    col_config = {
        "Ticker": st.column_config.TextColumn("Symbol"),
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Fair Value": st.column_config.NumberColumn("Graham Value", format="%.2f", help="Intrinsic Value based on Assets + Earnings"),
        "Safety Margin %": st.column_config.NumberColumn("Discount", format="%.1f%%"),
        "Trend": st.column_config.TextColumn("30d History"),
        "Volatility": st.column_config.NumberColumn("Risk", format="%.1%")
    }

    # FILTER & DISPLAY
    
    # 1. USA
    with tab1:
        st.subheader("🇺🇸 Undervalued US Stocks")
        # Filter: Only show stocks with valid Fair Value (>0) and sort by biggest discount
        df_us = df[(df["Region"] == "🇺🇸 USA") & (df["Fair Value"] > 0)].sort_values(by="Safety Margin %", ascending=False)
        st.dataframe(df_us.style.background_gradient(subset=['Safety Margin %'], cmap='RdYlGn', vmin=-20, vmax=50),
                     column_config=col_config, use_container_width=True, hide_index=True)

    # 2. BRAZIL
    with tab2:
        st.subheader("🇧🇷 Undervalued Brazilian Stocks (R$)")
        df_br = df[(df["Region"] == "🇧🇷 Brazil") & (df["Fair Value"] > 0)].sort_values(by="Safety Margin %", ascending=False)
        st.dataframe(df_br.style.background_gradient(subset=['Safety Margin %'], cmap='RdYlGn', vmin=-20, vmax=50),
                     column_config=col_config, use_container_width=True, hide_index=True)

    # 3. ETFs
    with tab3:
        st.subheader("🛡️ Global ETFs")
        df_etf = df[df["Region"] == "🛡️ ETF"]
        st.dataframe(df_etf, column_config=col_config, use_container_width=True, hide_index=True)

else:
    st.info("Click 'Run Max Capacity Scan' to begin. This will scan 600+ stocks.")
