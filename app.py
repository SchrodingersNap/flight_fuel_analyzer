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
        # Read the CSV file uploaded to the repo
        df = pd.read_csv('flight_fuel.csv')
        
        # Data Cleaning
        # Remove rows where 'qty' might be missing or non-numeric
        df = df.dropna(subset=['qty'])
        
        # Extract Carrier Code (first 2 letters usually) for grouping
        # This handles IDs like 'AI2706' -> 'AI'
        df['Carrier'] = df['flight_id'].astype(str).str[:2]
        return df
    except FileNotFoundError:
        st.error("⚠️ 'flight_fuel.csv' not found. Please upload it to the same directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

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
        
        analysis_mode = st.radio("Analyze by:", ["Specific Flight ID", "Carrier Code (AI, IX, 6E)"])
        
        if analysis_mode == "Specific Flight ID":
            flight_options = df['flight_id'].unique()
            # Default to a flight with data if possible
            default_idx = 0
            if 'AI2706' in flight_options:
                default_idx = list(flight_options).index('AI2706')
            
            selected_id = st.selectbox("Select Flight ID", flight_options, index=default_idx)
            filtered_df = df[df['flight_id'] == selected_id]
            title_text = f"Flight {selected_id}"
            
        else:
            carrier_options = df['Carrier'].unique()
            selected_id = st.selectbox("Select Carrier", carrier_options, index=0)
            filtered_df = df[df['Carrier'] == selected_id]
            title_text = f"Carrier {selected_id}"

        # Stats Calculation
        count = len(filtered_df)
        mean_val = filtered_df['qty'].mean()
        
        if count < 2:
            st.warning("⚠️ Not enough data points to calculate standard deviation. Showing raw value.")
            std_dev = 0.5 # Fallback for visualization
            p95 = mean_val
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
        
        # Create Normal Distribution Curve
        x_min = max(0, mean_val - 4*std_dev)
        x_max = mean_val + 4*std_dev
        x = np.linspace(x_min, x_max, 100)
        
        # Avoid division by zero if std_dev is 0
        safe_std = std_dev if std_dev > 0 else 0.1
        y = stats.norm.pdf(x, mean_val, safe_std)
        
        fig = go.Figure()
        
        # Add the Probability Density Curve
        fig.add_trace(go.Scatter(
            x=x, y=y, 
            mode='lines', 
            name='Probability Density', 
            fill='tozeroy', 
            line=dict(color='#4CAF50')
        ))
        
        # Add actual data points
        fig.add_trace(go.Scatter(
            x=filtered_df['qty'], 
            y=[0]*len(filtered_df), 
            mode='markers', 
            name='Observed Flights', 
            marker=dict(color='black', symbol='line-ns-open', size=10)
        ))
        
        fig.update_layout(
            xaxis_title="Fuel Qty (KL)",
            yaxis_title="Probability Density",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: BOWSER PROBABILITY CALCULATOR
# ==========================================
with tab2:
    st.header("🚛 Bowser Logistic Planning")
    st.markdown("Calculate probability that one Bowser can fuel multiple flights.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        bowser_cap = st.number_input("Bowser Fuel Capacity (KL)", min_value=1.0, value=20.0, step=0.5)
        
        st.markdown("### Select Flights")
        
        # Group stats by Carrier for simulation
        stats_db = df.groupby('Carrier')['qty'].agg(['mean', 'std']).reset_index()
        stats_db['std'] = stats_db['std'].fillna(0.5) 
        
        carriers = stats_db['Carrier'].unique()
        
        flight_a = st.selectbox("Flight A (Type)", carriers, index=0, key='fa')
        
        # Ensure we have enough carriers for defaults
        idx_b = 1 if len(carriers) > 1 else 0
        flight_b = st.selectbox("Flight B (Type)", carriers, index=idx_b, key='fb')
        
        flight_c = st.selectbox("Flight C (Optional)", ["None"] + list(carriers), key='fc')
        
    with col_result:
        # Get stats for Flight A
        row_a = stats_db[stats_db['Carrier'] == flight_a].iloc[0]
        mean_a, std_a = row_a['mean'], row_a['std']
        
        # Get stats for Flight B
        row_b = stats_db[stats_db['Carrier'] == flight_b].iloc[0]
        mean_b, std_b = row_b['mean'], row_b['std']
        
        total_mean = mean_a + mean_b
        total_var = (std_a**2) + (std_b**2)
        flights_list = [flight_a, flight_b]
        
        if flight_c != "None":
            row_c = stats_db[stats_db['Carrier'] == flight_c].iloc[0]
            mean_c, std_c = row_c['mean'], row_c['std']
            total_mean += mean_c
            total_var += (std_c**2)
            flights_list.append(flight_c)
            
        total_std = np.sqrt(total_var)
        
        # P(X <= Capacity)
        z_score = (bowser_cap - total_mean) / total_std
        probability = stats.norm.cdf(z_score)
        percentage = probability * 100
        
        st.markdown("### 📊 Prediction Results")
        st.write(f"**Flights:** {', '.join(flights_list)}")
        st.write(f"**Expected Total Fuel:** {total_mean:.2f} KL (± {total_std:.2f})")
        
        color = "red"
        if percentage > 80: color = "orange"
        if percentage > 95: color = "green"
        
        st.markdown(f"""
        <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; text-align: center;">
            <span style="font-size: 40px; color: {color}; font-weight: bold;">{percentage:.1f}%</span><br>
            <span style="font-size: 16px;">Success Probability</span>
        </div>
        """, unsafe_allow_html=True)
