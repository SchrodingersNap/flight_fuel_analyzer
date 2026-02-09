import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

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

df = load_data()
if df.empty: st.stop()

# --- HELPER: Smart Stats ---
def get_smart_stats(flight_id, dataframe):
    flight_data = dataframe[dataframe['flight_id'] == flight_id]
    mean_val = flight_data['qty'].mean()
    std_val = flight_data['qty'].std()
    
    # Smart Fallback
    if pd.isna(std_val) or len(flight_data) < 2:
        carrier_code = flight_id[:2]
        carrier_data = dataframe[dataframe['Carrier'] == carrier_code]
        std_val = carrier_data['qty'].std() if len(carrier_data) > 1 else 0.5
        
    return mean_val, std_val

# --- APP LAYOUT ---
st.title("✈️ Pro Flight Logistics & Risk Analyzer")

tab1, tab2 = st.tabs(["📊 Deep Dive Analysis", "🚛 Fleet Planning (Bowser)"])

# ==========================================
# TAB 1: DEEP DIVE ANALYSIS
# ==========================================
with tab1:
    col_config, col_main = st.columns([1, 3])
    
    with col_config:
        st.subheader("⚙️ Config")
        
        # Selection Logic
        analysis_mode = st.radio("Group By:", ["Specific Flight ID", "Carrier Code"], horizontal=True)
        
        if analysis_mode == "Specific Flight ID":
            options = sorted(df['flight_id'].unique())
            default_idx = options.index('IX1514') if 'IX1514' in options else 0
            selection = st.selectbox("Select Target:", options, index=default_idx)
            data_subset = df[df['flight_id'] == selection]
            name = f"Flight {selection}"
        else:
            options = sorted(df['Carrier'].unique())
            selection = st.selectbox("Select Target:", options, index=0)
            data_subset = df[df['Carrier'] == selection]
            name = f"Carrier {selection}"

        st.markdown("---")
        st.markdown("**Safety Margin**")
        confidence = st.slider("Confidence Level", 80, 99, 95, help="Higher % means carrying more fuel to be safer.")
        z_score = stats.norm.ppf(confidence / 100)

    with col_main:
        # --- CALCULATIONS ---
        # 1. Basic Stats
        if len(data_subset) < 2:
            # Handle single data point case using smart stats logic
            mean_val, std_dev = get_smart_stats(selection, df) if analysis_mode == "Specific Flight ID" else (data_subset['qty'].mean(), 0.5)
            max_val = mean_val # Can't know real max if only 1 point
        else:
            mean_val = data_subset['qty'].mean()
            std_dev = data_subset['qty'].std()
            max_val = data_subset['qty'].max()

        # 2. Advanced Metrics
        cv_score = (std_dev / mean_val) * 100  # Coefficient of Variation
        required_fuel = mean_val + (z_score * std_dev)
        buffer_needed = required_fuel - mean_val
        
        # --- UI: METRIC CARDS ---
        st.markdown(f"### Analysis for {name}")
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg. Consumption</div>
                <div class="metric-value">{mean_val:.2f} KL</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            # Color code the stability
            stab_color = "highlight-good" if cv_score < 15 else "highlight-bad"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Stability (CV)</div>
                <div class="metric-value {stab_color}">{cv_score:.1f}%</div>
                <small>{'✅ Stable' if cv_score < 15 else '⚠️ Volatile'}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Recorded</div>
                <div class="metric-value">{max_val:.2f} KL</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
             st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #4CAF50;">
                <div class="metric-label">Rec. Load ({confidence}%)</div>
                <div class="metric-value">{required_fuel:.2f} KL</div>
                <small>Buffer: +{buffer_needed:.2f} KL</small>
            </div>
            """, unsafe_allow_html=True)

        # --- VISUALIZATION: HYBRID CHART ---
        st.markdown("### 📉 Usage Distribution & Range")
        
        if mean_val > 0:
            fig = go.Figure()

            # 1. Normal Distribution Curve (The Theory)
            x_min = max(0, mean_val - 4*std_dev)
            x_max = mean_val + 4*std_dev
            x = np.linspace(x_min, x_max, 200)
            y = stats.norm.pdf(x, mean_val, std_dev)
            
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines', name='Prob. Curve', 
                fill='tozeroy', line=dict(color='rgba(31, 119, 180, 0.5)')
            ))

            # 2. Box Plot (The Reality) - Only if we have actual data points
            if len(data_subset) > 0:
                fig.add_trace(go.Box(
                    x=data_subset['qty'], 
                    name='Actual Data',
                    boxpoints='all', 
                    jitter=0.3, 
                    pointpos=-1.8,
                    marker=dict(color='black'),
                    line=dict(color='rgba(255, 99, 71, 0.8)')
                ))

            # 3. Add Line for Recommended Load
            fig.add_vline(x=required_fuel, line_dash="dash", line_color="green", annotation_text=f"Rec. Load ({confidence}%)")
            fig.add_vline(x=max_val, line_dash="dot", line_color="red", annotation_text="Max Rec.")

            fig.update_layout(
                height=400,
                xaxis_title="Fuel Quantity (KL)",
                yaxis_title="Probability Density",
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: BOWSER PLANNING (ENHANCED)
# ==========================================
with tab2:
    st.header("🚛 Multi-Flight Bowser Allocation")
    
    c_in, c_out = st.columns([1,1])
    
    with c_in:
        bowser_cap = st.number_input("Bowser Capacity (KL)", 10.0, 40.0, 20.0, 0.5)
        
        flights = sorted(df['flight_id'].unique())
        f1 = st.selectbox("Flight 1", flights, index=0)
        f2 = st.selectbox("Flight 2", flights, index=1 if len(flights)>1 else 0)
        f3 = st.selectbox("Flight 3 (Optional)", ["None"] + flights)
        
        # User defined risk tolerance
        risk_tol = st.slider("Acceptable Risk Level", 0.1, 10.0, 1.0, step=0.1, format="%f%%", help="1% means you accept running dry 1 in 100 times.")

    with c_out:
        # Calc Total Stats
        m1, s1 = get_smart_stats(f1, df)
        m2, s2 = get_smart_stats(f2, df)
        
        total_mean = m1 + m2
        total_var = s1**2 + s2**2
        
        if f3 != "None":
            m3, s3 = get_smart_stats(f3, df)
            total_mean += m3
            total_var += s3**2
            
        total_std = np.sqrt(total_var)
        
        # Calc Probability of Success (Capacity >= Demand)
        z_actual = (bowser_cap - total_mean) / total_std
        prob_success = stats.norm.cdf(z_actual) * 100
        risk_actual = 100 - prob_success
        
        # Visualization
        st.markdown("### 📊 Feasibility Report")
        
        if risk_actual <= risk_tol:
            status_color = "green"
            status_icon = "✅"
            status_msg = "APPROVED"
        elif risk_actual <= risk_tol * 2:
            status_color = "orange"
            status_icon = "⚠️"
            status_msg = "CAUTION"
        else:
            status_color = "red"
            status_icon = "❌"
            status_msg = "UNSAFE"

        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 10px solid {status_color};">
            <h2 style="color: {status_color}; margin:0;">{status_icon} {status_msg}</h2>
            <p style="font-size: 18px; margin: 5px 0;">Success Probability: <b>{prob_success:.2f}%</b></p>
            <p style="color: #666;">Risk of Shortage: {risk_actual:.2f}% (Target: <{risk_tol}%)</p>
            <hr>
            <p><b>Expected Total Demand:</b> {total_mean:.2f} KL</p>
            <p><b>Remaining Buffer:</b> {bowser_cap - total_mean:.2f} KL (Avg)</p>
        </div>
        """, unsafe_allow_html=True)
