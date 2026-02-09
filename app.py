import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Pro Flight Fuel Logistics", page_icon="✈️", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #000; }
    .metric-label { font-size: 14px; color: #666; margin-bottom: 5px; }
    .highlight-good { color: #28a745; }
    .highlight-bad { color: #dc3545; }
    .outlier-alert { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('flight_fuel.csv')
        df = df.dropna(subset=['qty'])
        df['Carrier'] = df['flight_id'].astype(str).str[:2]
        return df
    except FileNotFoundError:
        st.error("⚠️ 'flight_fuel.csv' not found.")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty: st.stop()

# --- HELPER: Outlier Removal ---
def remove_outliers(dataframe, threshold=3.0):
    """
    Removes data points with a Z-score higher than the threshold.
    Standard practice is threshold=3 (removes extreme 0.3% of data).
    """
    if len(dataframe) < 3: return dataframe, 0 # Too small to filter
    
    z_scores = np.abs(stats.zscore(dataframe['qty']))
    df_clean = dataframe[z_scores < threshold]
    removed_count = len(dataframe) - len(df_clean)
    return df_clean, removed_count

# --- HELPER: Smart Stats ---
def get_smart_stats(flight_id, dataframe):
    flight_data = dataframe[dataframe['flight_id'] == flight_id]
    
    # Handle empty after filtering
    if flight_data.empty: return 0, 0
        
    mean_val = flight_data['qty'].mean()
    std_val = flight_data['qty'].std()
    
    if pd.isna(std_val) or len(flight_data) < 2:
        carrier_code = flight_id[:2]
        carrier_data = dataframe[dataframe['Carrier'] == carrier_code]
        std_val = carrier_data['qty'].std() if len(carrier_data) > 1 else 0.5
        
    return mean_val, std_val

# --- APP LAYOUT ---
st.title("✈️ Pro Flight Logistics (Clean Data Mode)")

# --- GLOBAL SETTINGS SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Settings")
    enable_cleaning = st.checkbox("Remove Outliers", value=True, help="Removes extreme values that might distort planning.")
    
    if enable_cleaning:
        z_threshold = st.slider("Strictness (Z-Score)", 1.5, 4.0, 2.5, step=0.1, 
                                help="Lower value = More strict (removes more data). Standard is 3.0.")
        st.caption(f"Removing data > {z_threshold} Std Devs from mean.")
    
    st.divider()
    confidence = st.slider("Safety Confidence Level", 80, 99, 95)
    z_score_safety = stats.norm.ppf(confidence / 100)

# --- APPLY FILTER ---
if enable_cleaning:
    # We apply cleaning PER GROUP (Carrier or ID) later, or global here? 
    # Better to apply global cleaning contextually or per specific analysis to avoid removing valid variations between different flight types.
    # For this simple app, we will clean the specific subset selected in the tabs.
    pass 

tab1, tab2 = st.tabs(["📊 Deep Dive Analysis", "🚛 Fleet Planning (Bowser)"])

# ==========================================
# TAB 1: DEEP DIVE ANALYSIS
# ==========================================
with tab1:
    col_config, col_main = st.columns([1, 3])
    
    with col_config:
        st.subheader("Target Selection")
        analysis_mode = st.radio("Group By:", ["Specific Flight ID", "Carrier Code"])
        
        if analysis_mode == "Specific Flight ID":
            options = sorted(df_raw['flight_id'].unique())
            default_idx = options.index('IX1514') if 'IX1514' in options else 0
            selection = st.selectbox("Select Target:", options, index=default_idx)
            data_subset = df_raw[df_raw['flight_id'] == selection]
            name = f"Flight {selection}"
        else:
            options = sorted(df_raw['Carrier'].unique())
            selection = st.selectbox("Select Target:", options, index=0)
            data_subset = df_raw[df_raw['Carrier'] == selection]
            name = f"Carrier {selection}"

    with col_main:
        # --- 1. DATA CLEANING STEP ---
        original_count = len(data_subset)
        if enable_cleaning:
            data_clean, removed = remove_outliers(data_subset, z_threshold)
            if removed > 0:
                st.warning(f"🧹 Removed {removed} outlier(s) detected in this dataset. (Reduced from {original_count} to {len(data_clean)} records)")
            data_subset = data_clean # Override for calculations
        
        # --- 2. CALCULATIONS ---
        if len(data_subset) < 1:
            st.error("No data left after removing outliers! Try making the filter less strict.")
            st.stop()
            
        mean_val = data_subset['qty'].mean()
        
        # Smart Std Dev (if single point remains)
        if len(data_subset) < 2:
            std_dev = 0.5 # Default fallback
            if analysis_mode == "Specific Flight ID":
                _, std_dev = get_smart_stats(selection, df_raw) # Borrows from carrier
        else:
            std_dev = data_subset['qty'].std()
            
        max_val = data_subset['qty'].max()
        cv_score = (std_dev / mean_val) * 100
        required_fuel = mean_val + (z_score_safety * std_dev)
        buffer_needed = required_fuel - mean_val

        # --- 3. UI CARDS ---
        st.markdown(f"### Analysis for {name}")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.markdown(f"""<div class="metric-card"><div class="metric-label">Avg. Consumption</div><div class="metric-value">{mean_val:.2f} KL</div></div>""", unsafe_allow_html=True)
        
        stab_color = "highlight-good" if cv_score < 15 else "highlight-bad"
        m2.markdown(f"""<div class="metric-card"><div class="metric-label">Stability (CV)</div><div class="metric-value {stab_color}">{cv_score:.1f}%</div></div>""", unsafe_allow_html=True)
        
        m3.markdown(f"""<div class="metric-card"><div class="metric-label">Max (Cleaned)</div><div class="metric-value">{max_val:.2f} KL</div></div>""", unsafe_allow_html=True)

        m4.markdown(f"""<div class="metric-card" style="border: 2px solid #4CAF50;"><div class="metric-label">Rec. Load ({confidence}%)</div><div class="metric-value">{required_fuel:.2f} KL</div></div>""", unsafe_allow_html=True)

        # --- 4. VISUALIZATION ---
        st.markdown("### 📉 Cleaned Distribution")
        fig = go.Figure()

        # Normal Curve
        x_min = max(0, mean_val - 4*std_dev)
        x_max = mean_val + 4*std_dev
        x = np.linspace(x_min, x_max, 200)
        y = stats.norm.pdf(x, mean_val, std_dev)
        
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Prob. Curve', fill='tozeroy', line=dict(color='rgba(31, 119, 180, 0.5)')))

        # Actual Data Points (Rug Plot)
        fig.add_trace(go.Scatter(x=data_subset['qty'], y=[0]*len(data_subset), mode='markers', name='Valid Flights', marker=dict(color='black', symbol='line-ns-open', size=10)))

        # Recommended Load Line
        fig.add_vline(x=required_fuel, line_dash="dash", line_color="green", annotation_text=f"Rec. Load")

        fig.update_layout(height=400, xaxis_title="Fuel Quantity (KL)", yaxis_title="Probability Density", showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: BOWSER PLANNING
# ==========================================
with tab2:
    st.header("🚛 Bowser Allocation (Outlier-Free)")
    
    col_input, col_res = st.columns([1,1])
    
    with col_input:
        bowser_cap = st.number_input("Bowser Capacity (KL)", 10.0, 40.0, 20.0, 0.5)
        flights = sorted(df_raw['flight_id'].unique())
        f1 = st.selectbox("Flight 1", flights, index=0)
        f2 = st.selectbox("Flight 2", flights, index=1 if len(flights)>1 else 0)
        
    with col_res:
        # We need to apply cleaning to these specific flights before calculating stats
        def get_clean_stats(fid):
            # 1. Get Raw
            raw_data = df_raw[df_raw['flight_id'] == fid]
            # 2. Clean
            if enable_cleaning:
                clean_data, _ = remove_outliers(raw_data, z_threshold)
            else:
                clean_data = raw_data
            
            # 3. Calc Stats
            if len(clean_data) < 2:
                return get_smart_stats(fid, df_raw) # Fallback to smart stats
            return clean_data['qty'].mean(), clean_data['qty'].std()

        m1, s1 = get_clean_stats(f1)
        m2, s2 = get_clean_stats(f2)
        
        total_mean = m1 + m2
        total_std = np.sqrt(s1**2 + s2**2)
        
        # Calc Probability
        z_actual = (bowser_cap - total_mean) / total_std
        prob_success = stats.norm.cdf(z_actual) * 100
        
        st.markdown(f"**Total Expected Load:** {total_mean:.2f} KL")
        
        color = "green" if prob_success > 95 else "orange" if prob_success > 80 else "red"
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h1 style="color: {color}; margin:0;">{prob_success:.1f}%</h1>
            <p>Probability of Success</p>
        </div>
        """, unsafe_allow_html=True)
