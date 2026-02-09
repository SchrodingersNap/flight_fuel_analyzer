import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Flight Fuel Analyzer", page_icon="✈️", layout="wide")

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .metric-container { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load Data from CSV ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('flight_fuel.csv')
        df = df.dropna(subset=['qty'])
        # Create Carrier column for fallback statistics
        df['Carrier'] = df['flight_id'].astype(str).str[:2]
        return df
    except FileNotFoundError:
        st.error("⚠️ 'flight_fuel.csv' not found. Please upload it to the same directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- HELPER FUNCTION: Get Smart Stats ---
def get_smart_stats(flight_id, dataframe):
    """
    Returns (mean, std_dev) for a flight.
    If the flight has only 1 data point, it borrows the STD DEV from its Carrier 
    to provide a realistic risk assessment.
    """
    # 1. Get specific flight data
    flight_data = dataframe[dataframe['flight_id'] == flight_id]
    mean_val = flight_data['qty'].mean()
    std_val = flight_data['qty'].std()
    
    # 2. Smart Fallback if only 1 data point (std is NaN)
    if pd.isna(std_val):
        carrier_code = flight_id[:2]
        carrier_data = dataframe[dataframe['Carrier'] == carrier_code]
        # Use carrier volatility, or default to 0.5 if carrier also has 1 flight
        std_val = carrier_data['qty'].std() if len(carrier_data) > 1 else 0.5
        
    return mean_val, std_val

# --- HEADER ---
st.title("✈️ Flight Fuel Analyzer & Logistics")
st.markdown("Analyze historical fuel consumption and calculate logistics probabilities.")

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Single Flight Analysis", "🚛 Bowser Probability Calculator"])

# ==========================================
# TAB 1: SINGLE FLIGHT ANALYSIS
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # User explicitly asked for Individual Flights, but keeping the option is good practice
        analysis_mode = st.radio("Analyze by:", ["Specific Flight ID", "Carrier Code (AI, IX, 6E)"])
        
        if analysis_mode == "Specific Flight ID":
            flight_options = sorted(df['flight_id'].unique())
            default_idx = 0
            # Try to pick a flight with >1 entry for better demo
            if 'IX1514' in flight_options:
                default_idx = flight_options.index('IX1514')
            
            selected_id = st.selectbox("Select Flight ID", flight_options, index=default_idx)
            filtered_df = df[df['flight_id'] == selected_id]
            title_text = f"Flight {selected_id}"
            
        else:
            carrier_options = sorted(df['Carrier'].unique())
            selected_id = st.selectbox("Select Carrier", carrier_options, index=0)
            filtered_df = df[df['Carrier'] == selected_id]
            title_text = f"Carrier {selected_id}"

        # Stats Calculation for Chart
        count = len(filtered_df)
        mean_val = filtered_df['qty'].mean()
        
        if count < 2:
            st.info("ℹ️ Single data point detected. Calculating distribution using Carrier volatility.")
            # Use smart stats helper to fill the gap for the chart
            _, std_dev = get_smart_stats(filtered_df['flight_id'].iloc[0], df) if not filtered_df.empty else (0,0)
            p95 = mean_val + (1.645 * std_dev) # Estimate 95th percentile based on normal dist
        else:
            std_dev = filtered_df['qty'].std()
            p95 = np.percentile(filtered_df['qty'], 95)

        st.markdown("---")
        st.write(f"**Data Points:** {count}")

    with col2:
        st.subheader(f"Analysis: {title_text}")
        
        # 1. Stats Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Fuel", f"{mean_val:.2f} KL")
        c2.metric("Std Dev", f"{std_dev:.2f}")
        c3.metric("95th Percentile", f"{p95:.2f} KL")
        
        # 2. Distribution Plot
        st.markdown("#### Fuel Usage Distribution")
        
        if mean_val > 0:
            x_min = max(0, mean_val - 4*std_dev)
            x_max = mean_val + 4*std_dev
            x = np.linspace(x_min, x_max, 100)
            
            safe_std = std_dev if std_dev > 0 else 0.1
            y = stats.norm.pdf(x, mean_val, safe_std)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Probability Density', fill='tozeroy', line=dict(color='#4CAF50')))
            fig.add_trace(go.Scatter(x=filtered_df['qty'], y=[0]*len(filtered_df), mode='markers', name='Observed', marker=dict(color='black', symbol='line-ns-open', size=10)))
            
            fig.update_layout(xaxis_title="Fuel Qty (KL)", yaxis_title="Probability Density", showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available to plot.")

# ==========================================
# TAB 2: BOWSER PROBABILITY CALCULATOR (UPDATED)
# ==========================================
with tab2:
    st.header("🚛 Bowser Logistic Planning")
    st.markdown("Calculate probability that one Bowser can fuel **specific flights**.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        bowser_cap = st.number_input("Bowser Fuel Capacity (KL)", min_value=1.0, value=20.0, step=0.5)
        
        st.markdown("### Select Specific Flights")
        
        # Get all unique flight IDs sorted
        all_flights = sorted(df['flight_id'].unique())
        
        flight_a = st.selectbox("Flight A", all_flights, index=0, key='fa')
        
        # Try to set a different default for B
        idx_b = 1 if len(all_flights) > 1 else 0
        flight_b = st.selectbox("Flight B", all_flights, index=idx_b, key='fb')
        
        flight_c = st.selectbox("Flight C (Optional)", ["None"] + all_flights, key='fc')
        
    with col_result:
        # Calculate stats using the Smart Helper
        mean_a, std_a = get_smart_stats(flight_a, df)
        mean_b, std_b = get_smart_stats(flight_b, df)
        
        total_mean = mean_a + mean_b
        total_var = (std_a**2) + (std_b**2)
        flights_list = [flight_a, flight_b]
        
        if flight_c != "None":
            mean_c, std_c = get_smart_stats(flight_c, df)
            total_mean += mean_c
            total_var += (std_c**2)
            flights_list.append(flight_c)
            
        total_std = np.sqrt(total_var)
        
        # P(X <= Capacity)
        z_score = (bowser_cap - total_mean) / total_std
        probability = stats.norm.cdf(z_score)
        percentage = probability * 100
        
        st.markdown("### 📊 Prediction Results")
        st.write(f"**Selected:** {', '.join(flights_list)}")
        st.write(f"**Expected Load:** {total_mean:.2f} KL (± {total_std:.2f})")
        
        color = "red"
        if percentage > 80: color = "orange"
        if percentage > 95: color = "green"
        
        st.markdown(f"""
        <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color};">
            <span style="font-size: 40px; color: {color}; font-weight: bold;">{percentage:.1f}%</span><br>
            <span style="font-size: 16px;">Success Probability</span>
        </div>
        """, unsafe_allow_html=True)
        
        if percentage < 100:
             st.caption(f"*Note: Calculated using specific flight averages. Where data was scarce (1 flight), volatility was estimated using carrier ({flight_a[:2]}...) averages.*")
