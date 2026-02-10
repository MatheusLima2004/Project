import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import math
from datetime import datetime, timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Medallion Terminal", page_icon="🧬", layout="wide")
st.title("🧬 The Medallion Terminal (Ultimate AI)")
st.markdown("### 🧠 Fundamentals (Graham) + ⚡ Mean Reversion (Z-Score) + 🌊 TOC Squeeze")

# --- 2. USER GUIDE ---
with st.expander("📖 READ ME: The Strategy Guide", expanded=False):
    st.markdown("""
    ### 1. The "Medallion" Signals (Technical)
    * **Z-Score:** How rare is this price?
        * `> 2.0`: Statistically Expensive (Likely to drop/revert).
        * `< -2.0`: Statistically Cheap (Likely to bounce).
    * **TOC State (Constraint):**
        * `🔒 SQUEEZE`: Volatility is dead. A massive explosive move is imminent.
        * `🌊 EXPANSION`: The move is happening now.

    ### 2. The "Buffett" Signals (Fundamental)
    * **Graham Value:** The "true" worth of the company based on assets & earnings.
    * **Safety Margin:** Green means it's selling for less than it's worth.

    ### 3. The "Execution" (Options)
    * **Edge %:** The mathematical advantage of the option contract.
    * **Kelly %:** How much capital to risk.
    """)

# --- 3. MATH FORMULAS ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return max(S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2), 0)
        else:
            return max(K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1), 0)
    except: return 0

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

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    rf_us = st.number_input("🇺🇸 US Risk Free %", value=4.5) / 100
    rf_br = st.number_input("🇧🇷 BRL Risk Free %", value=11.25) / 100
    st.info("ℹ️ Scanning ~700 assets. Allow 3-4 minutes.")

# --- 5. DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_universe():
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        sp500 = [x.replace('.', '-') for x in sp500]
    except: sp500 = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]
    
    brazil = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "WEGE3.SA", "PRIO3.SA",
        "RENT3.SA", "SUZB3.SA", "GGBR4.SA", "JBSS3.SA", "CSAN3.SA", "BBSE3.SA", "ELET3.SA",
        "MELI", "NU", "PBR", "BSBR", "BBD"
    ]
    etfs = ["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "XLE", "XLF"]
    return list(set(sp500 + brazil + etfs))

@st.cache_data(ttl=3600)
def run_scan(rf_us, rf_br):
    tickers = get_universe()
    data_list = []
    
    progress_text = f"Crunching Numbers (Z-Scores, TOC, Black-Scholes)..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        hist_data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except: return pd.DataFrame()

    for i, ticker in enumerate(tickers):
        try:
            # 1. SETUP
            if ".SA" in ticker: r, region = rf_br, "🇧🇷 Brazil"
            elif ticker in ["SPY", "QQQ", "IWM"]: r, region = rf_us, "🛡️ ETF"
            else: r, region = rf_us, "🇺🇸 USA"

            df = hist_data[ticker] if len(tickers) > 1 else hist_data
            if df.empty or len(df) < 50: continue
            
            close = df['Close'].dropna()
            current_price = close.iloc[-1]
            spark = generate_sparkline(close)

            # 2. MEDALLION METRICS (Z-Score & TOC)
            # A. Z-Score (Mean Reversion)
            ma_50 = close.rolling(50).mean().iloc[-1]
            std_50 = close.rolling(50).std().iloc[-1]
            z_score = (current_price - ma_50) / std_50 if std_50 > 0 else 0
            
            # B. TOC (Bollinger Squeeze)
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            upper = sma_20 + (std_20 * 2)
            lower = sma_20 - (std_20 * 2)
            bandwidth = ((upper - lower) / sma_20) * 100
            bw_current = bandwidth.iloc[-1]
            bw_avg = bandwidth.rolling(50).mean().iloc[-1]
            
            toc_state = "Neutral"
            if bw_current < (bw_avg * 0.7): toc_state = "🔒 SQUEEZE" # High Potential Energy
            elif bw_current > (bw_avg * 1.5): toc_state = "🌊 EXPANSION"

            # 3. BUFFETT METRICS (Graham)
            fair_value = 0
            safety_margin = 0
            if region != "🛡️ ETF" and current_price > 5:
                try:
                    info = yf.Ticker(ticker).info
                    fair_value = calculate_graham_value(info.get('trailingEps', 0), info.get('bookValue', 0))
                    if fair_value > 0:
                        safety_margin = ((fair_value - current_price) / fair_value) * 100
                except: pass

            # 4. OPTIONS METRICS (Black-Scholes)
            best_opt, edge_pct, kelly_pct = "N/A", 0.0, 0.0
            
            # Smart Trigger: Only check options if there is a signal (Z-Score extreme or TOC Squeeze or Value)
            signal_active = (abs(z_score) > 2.0) or (toc_state == "🔒 SQUEEZE") or (safety_margin > 15)
            
            if signal_active and current_price > 5:
                try:
                    stock = yf.Ticker(ticker)
                    exps = stock.options
                    if exps:
                        target_date = exps[min(4, len(exps)-1)]
                        days = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.now()).days
                        T = days / 365
                        
                        # Direction: Z-Score < -2 (Oversold) -> Call. Z-Score > 2 (Overbought) -> Put.
                        if z_score < -1.5: opt_type = "call"
                        elif z_score > 1.5: opt_type = "put"
                        else: opt_type = "call" if current_price > ma_50 else "put" # Trend follow if no Z signal

                        chain = stock.option_chain(target_date)
                        opts = chain.calls if opt_type == "call" else chain.puts
                        opts = opts[(opts['strike'] > current_price * 0.95) & (opts['strike'] < current_price * 1.05)]
                        
                        if not opts.empty:
                            row = opts.iloc[0]
                            bs = black_scholes(current_price, row['strike'], T, r, std_50, opt_type) # Use hist vol
                            mkt = row['lastPrice']
                            if mkt > 0:
                                edge_pct = ((bs - mkt) / mkt) * 100
                                symbol = "C" if opt_type == "call" else "P"
                                best_opt = f"${row['strike']} {symbol}"
                                if edge_pct > 0: kelly_pct = kelly_criterion(0.55, 2.5) * 100
                except: pass

            data_list.append({
                "Ticker": ticker,
                "Region": region,
                "Price": current_price,
                "Z-Score": z_score,
                "TOC": toc_state,
                "Fair Value": fair_value,
                "Safety Margin %": safety_margin,
                "Best Option": best_opt,
                "Edge %": edge_pct,
                "Kelly %": kelly_pct,
                "Trend": spark
            })
            
        except: continue
        if i % 10 == 0: my_bar.progress(min((i + 1) / len(tickers), 1.0))
            
    my_bar.empty()
    return pd.DataFrame(data_list)

# --- 6. RENDER ---
if st.button("🚀 Run Medallion Scan"):
    st.cache_data.clear()

df = run_scan(rf_us, rf_br)

if not df.empty:
    tab1, tab2, tab3 = st.tabs(["🇺🇸 US Stocks", "🇧🇷 Brazil", "🛡️ ETFs"])
    
    cols = {
        "Ticker": st.column_config.TextColumn("Symbol"),
        "Price": st.column_config.NumberColumn("Price", format="%.2f"),
        "Z-Score": st.column_config.NumberColumn("Z-Score (Mean Rev)", format="%.2f"),
        "Safety Margin %": st.column_config.NumberColumn("Value Gap", format="%.1f%%"),
        "Edge %": st.column_config.NumberColumn("Opt Edge", format="%.1f%%"),
        "Kelly %": st.column_config.NumberColumn("Bet Size", format="%.1f%%"),
        "Trend": st.column_config.TextColumn("Chart")
    }

    with tab1:
        st.subheader("🇺🇸 US Opportunities")
        # Sort by Z-Score divergence (absolute value) to find anomalies
        df_us = df[df["Region"] == "🇺🇸 USA"].sort_values(by="Edge %", ascending=False)
        st.dataframe(df_us.style.background_gradient(subset=['Z-Score', 'Edge %'], cmap='RdYlGn', vmin=-3, vmax=3),
                     column_config=cols, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("🇧🇷 Brazil Opportunities")
        df_br = df[df["Region"] == "🇧🇷 Brazil"].sort_values(by="Edge %", ascending=False)
        st.dataframe(df_br.style.background_gradient(subset=['Z-Score', 'Safety Margin %'], cmap='RdYlGn', vmin=-3, vmax=3),
                     column_config=cols, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("🛡️ Global ETFs")
        df_etf = df[df["Region"] == "🛡️ ETF"]
        st.dataframe(df_etf, column_config=cols, use_container_width=True, hide_index=True)

else:
    st.info("Click 'Run Medallion Scan' to start. This searches for Fundamentals, Technical Anomalies (Z-Score), and Volatility Squeezes (TOC).")
