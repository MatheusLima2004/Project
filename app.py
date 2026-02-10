import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import math
from datetime import datetime, timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Ultimate Hedge Fund Terminal", page_icon="⚖️", layout="wide")
st.title("⚖️ The Ultimate Hedge Fund Terminal")
st.markdown("### 🌎 Global Scope | 📊 Graham Valuation | 🎯 Options Alpha")

# --- 2. USER GUIDE & DISCLAIMER ---
with st.expander("📖 HOW TO USE & METHODOLOGY (Click to Expand)", expanded=True):
    st.markdown("""
    ### 🎯 What is this tool for?
    This terminal is an **algorithmic scanner** designed to find mathematical discrepancies in the stock market. It does not "predict" the future; it identifies where assets are mispriced relative to their intrinsic value or statistical norms.

    ### ⚙️ The Mathematical Models
    1.  **Graham Fair Value (The "Value" Engine):**
        * Based on Benjamin Graham's formula: $V = \\sqrt{22.5 \\times EPS \\times BookValue}$.
        * It tells you what the company is worth based on its hard assets and earnings, ignoring hype.
    2.  **Black-Scholes Model (The "Options" Engine):**
        * Calculates the theoretical "Fair Price" of an option contract based on volatility and time.
        * If the Market Price is lower than this number, the option is statistically "On Sale."
    3.  **Kelly Criterion (The "Money Management" Engine):**
        * Calculates the exact percentage of your capital to risk to maximize growth while preventing ruin.

    ### ⚠️ RISK FREE RATE EXPLAINED
    * **Definition:** The theoretical return of an investment with zero risk (like a Government Bond).
    * **Why it matters:** It is the "gravity" of finance. Higher rates pull asset prices down.
    * **In this App:** We use **4.5% for US Stocks** (US Treasury) and **11.25% for Brazilian Stocks** (Selic Rate). This ensures the math is accurate for both economies.
    """)

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Market Settings")
    
    # DUAL RISK FREE RATES (Crucial for accuracy)
    col1, col2 = st.columns(2)
    with col1:
        rf_us = st.number_input("🇺🇸 US Rate %", value=4.5, help="US 10-Year Treasury Yield") / 100
    with col2:
        rf_br = st.number_input("🇧🇷 BRL Rate %", value=11.25, help="Brazil Selic Rate") / 100
        
    st.markdown("---")
    st.error("""
    **🚨 DISCLAIMER & LIABILITY**
    
    This software is for **educational and research purposes only**. 
    
    * **I am NOT a financial advisor.**
    * The signals generated are based on mathematical probabilities, not guarantees.
    * **You are 100% liable** for any financial losses incurred by using this data.
    * Options trading involves significant risk and is not suitable for all investors.
    """)

# --- 4. MATH FORMULAS ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        else:
            price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        return max(price, 0.0)
    except: return 0.0

def calculate_graham_value(eps, book_value):
    if eps > 0 and book_value > 0:
        return math.sqrt(22.5 * eps * book_value)
    return 0

def kelly_criterion(win_prob, win_loss_ratio):
    return max(0, (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio)

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

# --- 5. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_universe():
    # 1. S&P 500 (US)
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        sp500 = [x.replace('.', '-') for x in sp500]
    except:
        sp500 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V"]
        
    # 2. Brazil Top Liquid
    brazil = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "WEGE3.SA",
        "ABEV3.SA", "RENT3.SA", "BPAC11.SA", "SUZB3.SA", "HAPV3.SA", "RDOR3.SA",
        "B3SA3.SA", "EQTL3.SA", "LREN3.SA", "PRIO3.SA", "RAIL3.SA", "GGBR4.SA",
        "JBSS3.SA", "RADL3.SA", "VBBR3.SA", "CSAN3.SA", "TOTS3.SA", "BBSE3.SA",
        "MELI", "NU", "PBR", "BSBR", "BBD"
    ]
    # 3. ETFs
    etfs = ["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "XLE", "XLF"]
    
    return list(set(sp500 + brazil + etfs))

# --- 6. MAIN ENGINE ---
@st.cache_data(ttl=3600)
def run_scan(rf_us, rf_br):
    tickers = get_universe()
    data_list = []
    
    progress_text = f"Scanning {len(tickers)} Assets... (Please wait ~3 mins)"
    my_bar = st.progress(0, text=progress_text)
    
    try:
        hist_data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except: return pd.DataFrame()

    for i, ticker in enumerate(tickers):
        try:
            # 1. Determine Region & Rate
            if ".SA" in ticker:
                region = "🇧🇷 Brazil"
                r = rf_br
            elif ticker in ["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "XLE", "XLF"]:
                region = "🛡️ ETF"
                r = rf_us
            else:
                region = "🇺🇸 USA"
                r = rf_us

            # 2. Price Data
            df = hist_data[ticker] if len(tickers) > 1 else hist_data
            if df.empty or len(df) < 50: continue
            
            close = df['Close'].dropna()
            current_price = close.iloc[-1]
            
            # Trend & Volatility
            sma_50 = close.rolling(50).mean().iloc[-1]
            trend_dir = "🐂 Bull" if current_price > sma_50 else "🐻 Bear"
            spark = generate_sparkline(close)
            
            daily_returns = close.pct_change().dropna()
            hist_vol = daily_returns.std() * np.sqrt(252)

            # 3. Fair Value (Graham)
            fair_value = 0
            safety_margin = 0
            
            # Skip heavy info fetch for ETFs to save time
            if region != "🛡️ ETF" and current_price > 5:
                try:
                    info = yf.Ticker(ticker).info
                    eps = info.get('trailingEps', 0)
                    bvps = info.get('bookValue', 0)
                    fair_value = calculate_graham_value(eps, bvps)
                    
                    if fair_value > 0:
                        safety_margin = ((fair_value - current_price) / fair_value) * 100
                except: pass

            # 4. Options Engine
            best_contract = "N/A"
            edge_percent = 0.0
            kelly_pct = 0.0
            
            # Smart Filter: Check options if Volatile OR Cheap
            is_interesting = (safety_margin > 10 or hist_vol > 0.40 or region == "🇧🇷 Brazil")
            
            if is_interesting and current_price > 5:
                try:
                    stock = yf.Ticker(ticker)
                    exps = stock.options
                    if exps:
                        target_date = exps[min(4, len(exps)-1)]
                        days = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.now()).days
                        T = days / 365
                        
                        # Logic: Bull -> Calls, Bear -> Puts
                        opt_type = "call" if trend_dir == "🐂 Bull" else "put"
                        
                        chain = stock.option_chain(target_date)
                        opts = chain.calls if opt_type == "call" else chain.puts
                        opts = opts[(opts['strike'] > current_price * 0.95) & (opts['strike'] < current_price * 1.05)]
                        
                        if not opts.empty:
                            row = opts.iloc[0]
                            bs_price = black_scholes(current_price, row['strike'], T, r, hist_vol, opt_type)
                            mkt_price = row['lastPrice']
                            
                            if mkt_price > 0:
                                edge_percent = ((bs_price - mkt_price) / mkt_price) * 100
                                symbol = "C" if opt_type == "call" else "P"
                                best_contract = f"${row['strike']} {symbol} ({target_date})"
                                
                                if edge_percent > 0:
                                    kelly_pct = kelly_criterion(0.55, 2.5) * 100
                except: pass

            data_list.append({
                "Ticker": ticker,
                "Region": region,
                "Price": current_price,
                "Trend": spark,
                "Fair Value": fair_value,
                "Safety Margin %": safety_margin,
                "Direction": trend_dir,
                "Best Option": best_contract,
                "Edge %": edge_percent,
                "Kelly %": kelly_pct
            })
            
        except: continue
        if i % 10 == 0: my_bar.progress(min((i + 1) / len(tickers), 1.0))
            
    my_bar.empty()
    return pd.DataFrame(data_list)

# --- 7. RENDER ---
if st.button("🚀 Run Full Analysis"):
    st.cache_data.clear()

df = run_scan(rf_us, rf_br)

if not df.empty:
    tab1, tab2, tab3 = st.tabs(["🇺🇸 US Stocks", "🇧🇷 Brazil Stocks", "🛡️ ETFs"])
    
    col_config = {
        "Ticker": st.column_config.TextColumn("Symbol"),
        "Price": st.column_config.NumberColumn("Price", format="%.2f"),
        "Fair Value": st.column_config.NumberColumn("Graham Value", format="%.2f"),
        "Safety Margin %": st.column_config.NumberColumn("Safety Margin", format="%.1f%%"),
        "Edge %": st.column_config.NumberColumn("Option Edge", format="%.1f%%"),
        "Kelly %": st.column_config.NumberColumn("Bet Size", format="%.1f%%"),
        "Trend": st.column_config.TextColumn("30d History")
    }
    
    with tab1:
        st.subheader("🇺🇸 US Opportunities (USD)")
        df_us = df[df["Region"] == "🇺🇸 USA"].sort_values(by="Edge %", ascending=False)
        st.dataframe(df_
