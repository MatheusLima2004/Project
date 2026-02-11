# --- 1. UPDATED SOVEREIGN ENGINE ---
class SovereignEngine:
    @staticmethod
    def get_tickers():
        try:
            # Mining the full S&P 500 node
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return sp500['Symbol'].tolist()
        except:
            return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "SPY", "QQQ"]

    @staticmethod
    def multi_node_mine(ticker_list):
        """Bypasses the 4-stock limit using Staggered Chunks"""
        all_data = {}
        chunk_size = 20 # Nodes of 20 provide the best stability/speed ratio
        
        # FIX: Move progress tracking to the main area to avoid API Exceptions
        progress_text = st.empty()
        progress_bar = st.progress(0)

        for i in range(0, len(ticker_list), chunk_size):
            chunk = ticker_list[i:i + chunk_size]
            current_node = i + len(chunk)
            
            progress_text.markdown(f"📡 **Sovereign Mining:** Node {current_node} of {len(ticker_list)} assets...")
            
            # The Multi-Node Download
            data = yf.download(chunk, period="1y", interval="1d", group_by='ticker', threads=True, progress=False)
            
            for t in chunk:
                if t in data and not data[t].dropna().empty:
                    all_data[t] = data[t]
            
            # Update Progress Bar (Ensures value stays between 0.0 and 1.0)
            percent = min(current_node / len(ticker_list), 1.0)
            progress_bar.progress(percent)
            
            # The Sovereign Pulse (Randomized delay to bypass security blocks)
            time.sleep(random.uniform(1.2, 2.8))
            
        # Clean up progress UI after mining is complete
        progress_text.empty()
        progress_bar.empty()
        return all_data

# --- 2. UPDATED UI RENDER ---
@st.fragment(run_every=900) # 15-Minute Sovereign Cycle
def render_terminal():
    st.title("💹 Sovereign Matrix | Institutional Multi-Node")
    
    engine = SovereignEngine()
    # Scrape the full market tickers
    ticker_list = engine.get_tickers()
    
    with st.status("Initializing Distributed Node Clusters...", expanded=True) as status:
        data_mesh = engine.multi_node_mine(ticker_list)
        status.update(label="Deep Scan Complete. Vectorizing Algorithms...", state="complete", expanded=False)
        
        # Matrix Math (Z-Score & PoP)
        results = []
        for t, df in data_mesh.items():
            try:
                # Medallion-grade feature extraction
                curr = df['Close'].iloc[-1]
                z = (curr - df['Close'].rolling(50).mean().iloc[-1]) / df['Close'].rolling(50).std().iloc[-1]
                sigma = df['Close'].pct_change().std() * np.sqrt(252)
                results.append({"Symbol": t, "Price": curr, "Z-Score": z, "Volatility": sigma})
            except: continue

    if results:
        df_final = pd.DataFrame(results)
        st.subheader("📊 Global Command Matrix")
        st.dataframe(
            df_final.style.background_gradient(subset=['Z-Score'], cmap='RdYlGn_r')
            .format({"Price": "${:.2f}", "Z-Score": "{:.2f}", "Volatility": "{:.1%}"}),
            use_container_width=True, hide_index=True, height=600
        )
    else:
        st.error("Protocol Failure: Node Cluster Blocked. IP Refresh required.")
