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

# --- 1. Load Data with Strict Filtering & Space Normalization ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('flight_fuel.csv')
        df = df.dropna(subset=['qty'])
        
        # --- NEW: NORMALIZE FLIGHT IDs ---
        # Convert to string, make uppercase, and remove ALL spaces
        # This makes "6E 114", "6e 114", and "6E114" identical
        df['flight_id'] = df['flight_id'].astype(str).str.upper().str.replace(" ", "", regex=False)
        
        # Create Carrier Code column
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
    flight_data = dataframe[dataframe['flight_id'] == flight_id]
    if flight_data.empty: return 0, 0
    
    mean_val = flight_data['qty'].mean()
    std_val = flight_data['qty'].std()
    
    if pd.isna(std_val) or len(flight_data) < 2:
        carrier_code = flight_id[:2]
        carrier_data = dataframe[dataframe['Carrier'] == carrier_code]
        std_val = carrier_data['qty'].std() if len(carrier_data) > 1 else 0.5
        
    return mean_val, std_val


# --- APP LAYOUT ---
st.title("✈️ Pro Flight Logistics")
st.markdown(f"**Data Status:** Loaded {valid_rows} valid flights (Filtered out {total_rows - valid_rows} unknown carriers).")

# --- GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ Data Settings")
    enable_cleaning = st.checkbox("Remove Outliers", value=True, help="Removes extreme values (>3 Std Dev).")
    
    if enable_cleaning:
        z_threshold = st.slider("Strictness (Z-Score)", 1.5, 4.0, 3.0, step=0.1)
    
    st.divider()
    confidence = st.slider("Confidence Level", 80, 99, 95)
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
        # 1. Clean Data
        if enable_cleaning:
            data_clean, removed = remove_outliers(data_subset, z_threshold)
            if removed > 0:
                st.caption(f"🧹 Removed {removed} outliers.")
            data_subset = data_clean 
        
        if len(data_subset) < 1:
            st.warning("No data available.")
            st.stop()
            
        # 2. Calc Stats
        mean_val = data_subset['qty'].mean()
        
        if len(data_subset) < 2:
            std_dev = 0.5 
            if analysis_mode == "Specific Flight ID":
                _, std_dev = get_smart_stats(selection, df_raw) 
        else:
            std_dev = data_subset['qty'].std()
            
        max_val = data_subset['qty'].max()
        cv_score = (std_dev / mean_val) * 100
        required_fuel = mean_val + (z_score_safety * std_dev)

        # 3. Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""<div class="metric-card"><div class="metric-label">Avg. Consumption</div><div class="metric-value">{mean_val:.2f} KL</div></div>""", unsafe_allow_html=True)
        stab_color = "highlight-good" if cv_score < 15 else "highlight-bad"
        m2.markdown(f"""<div class="metric-card"><div class="metric-label">Stability (CV)</div><div class="metric-value {stab_color}">{cv_score:.1f}%</div></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card"><div class="metric-label">Max Recorded</div><div class="metric-value">{max_val:.2f} KL</div></div>""", unsafe_allow_html=True)
        m4.markdown(f"""<div class="metric-card" style="border: 2px solid #4CAF50;"><div class="metric-label">Rec. Load ({confidence}%)</div><div class="metric-value">{required_fuel:.2f} KL</div></div>""", unsafe_allow_html=True)

        # 4. Chart
        fig = go.Figure()
        x_min = max(0, mean_val - 4*std_dev)
        x_max = mean_val + 4*std_dev
        x = np.linspace(x_min, x_max, 200)
        y = stats.norm.pdf(x, mean_val, std_dev)
        
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Prob. Curve', fill='tozeroy', line=dict(color='rgba(31, 119, 180, 0.5)')))
        fig.add_trace(go.Scatter(x=data_subset['qty'], y=[0]*len(data_subset), mode='markers', name='Valid Flights', marker=dict(color='black', symbol='line-ns-open', size=10)))
        fig.add_vline(x=required_fuel, line_dash="dash", line_color="green", annotation_text=f"Rec. Load")

        fig.update_layout(height=400, xaxis_title="Fuel Quantity (KL)", yaxis_title="Probability Density", showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: BOWSER ALLOCATION
# ==========================================
with tab2:
    st.header("🚛 Bowser Allocation")
    
    col_input, col_res = st.columns([1,1])
    
    with col_input:
        bowser_cap = st.number_input("Bowser Capacity (KL)", 0.0, 50.0, 15.0, 0.5)
        # Only show flights that survived the carrier filter
        flights = sorted(df_raw['flight_id'].unique())
        f1 = st.selectbox("Flight 1", flights, index=0)
        f2 = st.selectbox("Flight 2", flights, index=1 if len(flights)>1 else 0)
        
    with col_res:
        def get_clean_stats(fid):
            raw_data = df_raw[df_raw['flight_id'] == fid]
            if enable_cleaning:
                clean_data, _ = remove_outliers(raw_data, z_threshold)
            else:
                clean_data = raw_data
            
            if len(clean_data) < 2:
                return get_smart_stats(fid, df_raw) 
            return clean_data['qty'].mean(), clean_data['qty'].std()

        m1, s1 = get_clean_stats(f1)
        m2, s2 = get_clean_stats(f2)
        
        total_mean = m1 + m2
        total_std = np.sqrt(s1**2 + s2**2)
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
