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
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load Data with ROBUST Normalization ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('flight_fuel.csv')
        df = df.dropna(subset=['qty'])
        
        # --- NORMALIZE FLIGHT IDs (The Fix) ---
        # 1. Convert to string & Uppercase
        # 2. Remove Spaces (" ")
        # 3. Remove Hyphens ("-")
        # Now "6E-2248" and "6E 2248" both become "6E2248"
        df['flight_id'] = df['flight_id'].astype(str).str.upper().str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
        
        # Create Carrier Code column (first 2 chars)
        df['Carrier'] = df['flight_id'].str[:2]
        
        # --- STRICT CARRIER FILTER ---
        allowed_carriers = [
            '6E', 'AI', 'IX', '9I', 'MH', 'QP', 'SQ', 'AK', 'FD', 'TG', 
            'VZ', 'EK', 'BS', 'KB', 'B3', 'FZ', 'NS', 'U4', 'SL'
        ]
        
        original_count = len(df)
        df = df[df['Carrier'].isin(allowed_carriers)]
        filtered_count = len(df)
        
        return df, original_count, filtered_count
        
    except FileNotFoundError:
        return pd.DataFrame(), 0, 0

# Load the data
df_raw, total_rows, valid_rows = load_data()

if df_raw.empty:
    st.error("⚠️ 'flight_fuel.csv' not found or no valid data matching your carrier list.")
    st.stop()

# --- HELPER: Outlier Removal ---
def remove_outliers(dataframe, threshold=3.0):
    if len(dataframe) < 3: return dataframe, 0
    z_scores = np.abs(stats.zscore(dataframe['qty']))
    df_clean = dataframe[z_scores < threshold]
    removed_count = len(dataframe) - len(df_clean)
    return df_clean, removed_count

# --- HELPER: Smart Stats ---
def get_smart_stats(flight_id, dataframe):
    """
    Returns (mean, std_dev) for a flight.
    Falls back to Carrier Stats if flight has insufficient data.
    """
    flight_data = dataframe[dataframe['flight_id'] == flight_id]
    if flight_data.empty: return 0, 0
    
    mean_val = flight_data['qty'].mean()
    std_val = flight_data['qty'].std()
    
    # Fallback logic
    if pd.isna(std_val) or len(flight_data) < 2:
        carrier_code = flight_id[:2]
        carrier_data = dataframe[dataframe['Carrier'] == carrier_code]
        std_val = carrier_data['qty'].std() if len(carrier_data) > 1 else 0.5
        
    return mean_val, std_val


# --- APP LAYOUT ---
st.title("✈️ Pro Flight Logistics")
st.markdown(f"**Data Status:** Loaded {valid_rows} valid flights (Filtered out {total_rows - valid_rows} unknown/invalid rows).")

# --- GLOBAL SETTINGS SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Settings")
    enable_cleaning = st.checkbox("Remove Outliers", value=True, help="Removes extreme values (>3 Std Dev).")
    
    if enable_cleaning:
        z_threshold = st.slider("Strictness (Z-Score)", 1.5, 4.0, 3.0, step=0.1)
    
    st.divider()
    confidence = st.slider("Single Flight Confidence", 80, 99, 95)
    z_score_safety = stats.norm.ppf(confidence / 100)

tab1, tab2 = st.tabs(["📊 Single Flight Analysis", "🚛 Bowser Allocation"])

# ==========================================
# TAB 1: SINGLE FLIGHT ANALYSIS
# ==========================================
with tab1:
    col_config, col_main = st.columns([1, 3])
    
    with col_config:
        st.subheader("Selection")
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
        # 1. Clean Data (Local to this tab)
        if enable_cleaning:
            data_clean, removed = remove_outliers(data_subset, z_threshold)
            if removed > 0:
                st.caption(f"🧹 Removed {removed} outliers from analysis.")
            data_subset = data_clean 
        
        if len(data_subset) < 1:
            st.warning("No data available after filtering.")
            st.stop()
            
        # 2. Calc Stats
        mean_val = data_subset['qty'].mean()
        
        # Handle Low Data Count
        if len(data_subset) < 2:
            std_dev = 0.5 
            if analysis_mode == "Specific Flight ID":
                # Use smart stats fallback
                _, std_dev = get_smart_stats(selection, df_raw) 
        else:
            std_dev = data_subset['qty'].std()
            
        max_val = data_subset['qty'].max()
        cv_score = (std_dev / mean_val) * 100
        required_fuel = mean_val + (z_score_safety * std_dev)

        # 3. Metric Cards
        st.markdown(f"### Analysis for {name}")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.markdown(f"""<div class="metric-card"><div class="metric-label">Avg. Consumption</div><div class="metric-value">{mean_val:.2f} KL</div></div>""", unsafe_allow_html=True)
        
        stab_color = "highlight-good" if cv_score < 15 else "highlight-bad"
        m2.markdown(f"""<div class="metric-card"><div class="metric-label">Stability (CV)</div><div class="metric-value {stab_color}">{cv_score:.1f}%</div></div>""", unsafe_allow_html=True)
        
        m3.markdown(f"""<div class="metric-card"><div class="metric-label">Max Recorded</div><div class="metric-value">{max_val:.2f} KL</div></div>""", unsafe_allow_html=True)
        
        m4.markdown(f"""<div class="metric-card" style="border: 2px solid #4CAF50;"><div class="metric-label">Rec. Load ({confidence}%)</div><div class="metric-value">{required_fuel:.2f} KL</div></div>""", unsafe_allow_html=True)

        # 4. Visualization Chart
        fig = go.Figure()
        
        # Create Normal Distribution Curve
        x_min = max(0, mean_val - 4*std_dev)
        x_max = mean_val + 4*std_dev
        x = np.linspace(x_min, x_max, 200)
        y = stats.norm.pdf(x, mean_val, std_dev)
        
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Prob. Curve', fill='tozeroy', line=dict(color='rgba(31, 119, 180, 0.5)')))
        
        # Add Actual Data Points
        fig.add_trace(go.Scatter(x=data_subset['qty'], y=[0]*len(data_subset), mode='markers', name='Valid Flights', marker=dict(color='black', symbol='line-ns-open', size=10)))
        
        # Add Reference Line
        fig.add_vline(x=required_fuel, line_dash="dash", line_color="green", annotation_text=f"Rec. Load")

        fig.update_layout(height=400, xaxis_title="Fuel Quantity (KL)", yaxis_title="Probability Density", showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: BOWSER ALLOCATION (COST OPTIMIZED)
# ==========================================
with tab2:
    st.header("🚛 Bowser Allocation (Cost Optimized)")
    
    col_input, col_res = st.columns([1,1])
    
    with col_input:
        bowser_cap = st.number_input("Bowser Capacity (KL)", 0.0, 56.0, 15.0, 0.5)
        
        # Filter for valid carriers only
        flights = sorted(df_raw['flight_id'].unique())
        f1 = st.selectbox("Flight 1", flights, index=0)
        f2 = st.selectbox("Flight 2", flights, index=1 if len(flights)>1 else 0)
        f3 = st.selectbox("Flight 3 (Optional)", ["None"] + flights)
        
        st.divider()
        # Default set to 95% for cost efficiency
        fleet_confidence = st.slider("Fleet Safety Confidence", 90, 99, 95, key="fleet_conf", 
                                   help="95% is industry standard. 99% is expensive (carrying too much buffer).")
        z_fleet = stats.norm.ppf(fleet_confidence / 100)
        
    with col_res:
        # Helper to get clean stats for planning
        def get_clean_stats(fid):
            raw_data = df_raw[df_raw['flight_id'] == fid]
            if enable_cleaning:
                clean_data, _ = remove_outliers(raw_data, z_threshold)
            else:
                clean_data = raw_data
            
            # Use smart stats if data is too thin
            if len(clean_data) < 2:
                return get_smart_stats(fid, df_raw) 
            return clean_data['qty'].mean(), clean_data['qty'].std()

        # 1. Calculate Individual Stats
        m1, s1 = get_clean_stats(f1)
        m2, s2 = get_clean_stats(f2)
        
        total_mean = m1 + m2
        total_var = s1**2 + s2**2
        
        if f3 != "None":
            m3, s3 = get_clean_stats(f3)
            total_mean += m3
            total_var += s3**2
            
        total_std = np.sqrt(total_var)
        
        # 2. Calculate "Optimized Load" for the WHOLE FLEET
        rec_fleet_load = total_mean + (z_fleet * total_std)
        
        # 3. Calculate Probability of current Bowser working
        z_actual = (bowser_cap - total_mean) / total_std
        prob_success = stats.norm.cdf(z_actual) * 100
        
        # --- DISPLAY RESULTS ---
        st.subheader("📊 Planning Results")
        
        # Card 1: The Probability
        color = "green" if prob_success >= fleet_confidence else "orange" if prob_success > 80 else "red"
        
        st.markdown(f"""
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 20px;">
            <strong style="font-size: 18px;">Can the {bowser_cap} KL Bowser do it?</strong>
            <h2 style="color: {color}; margin: 5px 0;">{prob_success:.1f}% Probability</h2>
            <small>Target: {fleet_confidence}%</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Card 2: The Optimized Load
        diff = bowser_cap - rec_fleet_load
        
        if diff >= 0:
            status = f"✅ Fits (Spare: {diff:.2f} KL)"
            status_color = "#28a745"
            bg_color = "#e6fffa"
        else:
            status = f"❌ Short by {abs(diff):.2f} KL"
            status_color = "#dc3545"
            bg_color = "#fff5f5"
        
        st.markdown(f"""
        <div style="padding: 15px; background-color: {bg_color}; border-radius: 10px; border-left: 5px solid {status_color};">
            <strong style="color: #333;">Optimized Fuel Required ({fleet_confidence}%)</strong>
            <div style="font-size: 28px; font-weight: bold; color: #333;">{rec_fleet_load:.2f} KL</div>
            <div style="color: {status_color}; font-weight: bold; margin-top: 5px;">{status}</div>
            <hr style="margin: 10px 0; border-top: 1px dashed #ccc;">
            <div style="display: flex; justify-content: space-between; font-size: 14px; color: #666;">
                <span>Avg Demand: <b>{total_mean:.2f}</b></span>
                <span>Safety Buffer: <b>+{rec_fleet_load - total_mean:.2f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
