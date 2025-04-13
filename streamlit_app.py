import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Industrial Equipment Monitoring & RUL Prediction", layout="wide")

# Global variables
df = None
thresholds = {}
thresholds_rul = {}
operating_hours_per_day = 16
estimated_total_lifespan_hours = 5000
maintenance_threshold = 25  # RUL percentage threshold for maintenance requirement
current_date = datetime(2025, 4, 13)  # Current date: April 13, 2025
data_source = "Sample"  # To track whether we're using sample or uploaded data

# Function to generate sample data if no file is uploaded
def generate_sample_data():
    global df, thresholds, thresholds_rul, data_source
    
    # Create a timestamp range for the last 180 days
    start_date = current_date - timedelta(days=180)
    timestamps = pd.date_range(start=start_date, end=current_date, freq='D')
    n_samples = len(timestamps)
    
    # Generate base values with some randomness
    np.random.seed(42)  # For reproducibility
    
    # Create initial healthy values
    base_temp = 70
    base_x_rms_vel = 0.2
    base_z_rms_vel = 0.18
    base_x_peak_vel = 0.3
    base_z_peak_vel = 0.28
    base_x_rms_accel = 2.0
    base_z_rms_accel = 1.8
    base_x_peak_accel = 3.0
    base_z_peak_accel = 2.8
    
    # Create degradation patterns over time
    # More severe degradation starts around day 120
    days = np.arange(n_samples)
    degradation = np.ones(n_samples)
    degradation[120:] = np.linspace(1, 1.8, n_samples-120)  # Increasing degradation
    
    # Add some weekly cyclical patterns 
    weekly_cycle = 0.1 * np.sin(days * 2 * np.pi / 7)
    
    # Generate the data with noise and trends
    temperature = base_temp + degradation * 5 + weekly_cycle + np.random.normal(0, 2, n_samples)
    x_rms_vel = base_x_rms_vel + degradation * 0.05 + 0.02 * weekly_cycle + np.random.normal(0, 0.01, n_samples)
    z_rms_vel = base_z_rms_vel + degradation * 0.04 + 0.02 * weekly_cycle + np.random.normal(0, 0.01, n_samples)
    x_peak_vel = base_x_peak_vel + degradation * 0.07 + 0.03 * weekly_cycle + np.random.normal(0, 0.02, n_samples)
    z_peak_vel = base_z_peak_vel + degradation * 0.06 + 0.03 * weekly_cycle + np.random.normal(0, 0.02, n_samples)
    x_rms_accel = base_x_rms_accel + degradation * 0.3 + 0.1 * weekly_cycle + np.random.normal(0, 0.1, n_samples)
    z_rms_accel = base_z_rms_accel + degradation * 0.25 + 0.1 * weekly_cycle + np.random.normal(0, 0.1, n_samples)
    x_peak_accel = base_x_peak_accel + degradation * 0.4 + 0.15 * weekly_cycle + np.random.normal(0, 0.2, n_samples)
    z_peak_accel = base_z_peak_accel + degradation * 0.35 + 0.15 * weekly_cycle + np.random.normal(0, 0.2, n_samples)
    
    # Create the DataFrame
    data = {
        'timestamp': timestamps,
        'temperature': temperature,
        'x_rms_vel': x_rms_vel,
        'z_rms_vel': z_rms_vel,
        'x_peak_vel': x_peak_vel,
        'z_peak_vel': z_peak_vel,
        'x_rms_accel': x_rms_accel,
        'z_rms_accel': z_rms_accel,
        'x_peak_accel': x_peak_accel, 
        'z_peak_accel': z_peak_accel
    }
    
    df = pd.DataFrame(data)
    data_source = "Sample"
    
    calculate_thresholds()
    process_data()
    return df

# Process data function to apply all the calculations
def process_data():
    global df
    
    window_size = 30
    # Calculate rolling means for trend analysis
    df['temp_mean'] = df['temperature'].rolling(window=window_size).mean()
    df['x_rms_vel_mean'] = df['x_rms_vel'].rolling(window=window_size).mean()
    df['z_rms_vel_mean'] = df['z_rms_vel'].rolling(window=window_size).mean()
    df['x_rms_accel_mean'] = df['x_rms_accel'].rolling(window=window_size).mean()
    df['z_rms_accel_mean'] = df['z_rms_accel'].rolling(window=window_size).mean()

    # Set alert flags based on thresholds
    df['temp_alert'] = df['temperature'] > thresholds['temperature']
    df['x_rms_vel_alert'] = df['x_rms_vel'] > thresholds['x_rms_vel']
    df['z_rms_vel_alert'] = df['z_rms_vel'] > thresholds['z_rms_vel']
    df['x_rms_accel_alert'] = df['x_rms_accel'] > thresholds['x_rms_accel']
    df['z_rms_accel_alert'] = df['z_rms_accel'] > thresholds['z_rms_accel']

    def estimate_rul(data, threshold):
        return np.maximum(0, (1 - (data / threshold)) * 100)

    # Calculate RUL for each parameter
    for param in thresholds_rul:
        if param in df.columns:
            df[f'{param}_rul'] = estimate_rul(df[param], thresholds_rul[param])

    # Calculate a combined health index (weighted average of all parameters)
    # Higher weights for acceleration parameters as they often indicate developing issues
    weights = {
        'temperature_rul': 0.1,
        'x_rms_vel_rul': 0.1, 
        'z_rms_vel_rul': 0.1,
        'x_peak_vel_rul': 0.1,
        'z_peak_vel_rul': 0.1,
        'x_rms_accel_rul': 0.15,
        'z_rms_accel_rul': 0.15,
        'x_peak_accel_rul': 0.15,
        'z_peak_accel_rul': 0.15
    }
    
    # Make sure all columns exist before calculating weighted health index
    valid_columns = [col for col in weights.keys() if col in df.columns]
    if valid_columns:
        df['health_index'] = sum(df[col] * weights[col] for col in valid_columns) / sum(weights[col] for col in valid_columns)

# Load dataset function with file upload option
def load_dataset(uploaded_file):
    global df, thresholds, thresholds_rul, data_source, current_date
    try:
        # Read the Excel file directly with column names from first row
        df = pd.read_excel(uploaded_file)
        
        # Rename columns to standardized format for internal processing
        column_mapping = {
            "TimeStamp": "timestamp",
            "Temperature": "temperature", 
            "X_RMS_Vel": "x_rms_vel",
            "Z_RMS_Vel": "z_rms_vel",
            "X_Peak_Vel": "x_peak_vel", 
            "Z_Peak_Vel": "z_peak_vel",
            "X_RMS_Accel": "x_rms_accel",
            "Z_RMS_Accel": "z_rms_accel",
            "X_Peak_Accel": "x_peak_accel",
            "Z_Peak_Accel": "z_peak_accel"
        }
        
        # Convert all column names to strings to handle numeric column indices
        df.columns = df.columns.astype(str)
        
        # Rename the columns that exist in the mapping
        renamed_columns = {}
        for old_col, new_col in column_mapping.items():
            for col in df.columns:
                if old_col.lower() in col.lower():
                    renamed_columns[col] = new_col
        
        df.rename(columns=renamed_columns, inplace=True)
        
        # Make sure required columns exist
        required_columns = ["timestamp", "temperature", "x_rms_vel", "z_rms_vel"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return f"Error: Missing required columns: {', '.join(missing_columns)}"
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna(subset=["timestamp"])  # Remove rows with missing timestamps
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')  # Forward-fill then backward-fill
        
        # Set current date to the latest timestamp in the data if available
        if not df.empty:
            current_date = df["timestamp"].max()
        
        # Mark as uploaded data
        data_source = "Uploaded"
        
        # Calculate thresholds and process data
        calculate_thresholds()
        process_data()

        return f"Data loaded successfully! Dataset contains {len(df)} samples from {df['timestamp'].min().strftime('%B %d, %Y')} to {df['timestamp'].max().strftime('%B %d, %Y')}"
    except Exception as e:
        return f"Error loading data: {str(e)}"

def calculate_thresholds():
    global df, thresholds, thresholds_rul
    
    # Calculate thresholds for all parameters that exist in the dataframe
    threshold_columns = [
        'temperature', 'x_rms_vel', 'z_rms_vel', 'x_peak_vel', 'z_peak_vel',
        'x_rms_accel', 'z_rms_accel', 'x_peak_accel', 'z_peak_accel'
    ]
    
    thresholds = {}
    for col in threshold_columns:
        if col in df.columns:
            thresholds[col] = np.percentile(df[col].dropna(), 95)
    
    # Define critical thresholds for RUL calculation
    # Use default values that can be overridden based on domain knowledge
    default_thresholds_rul = {
        'temperature': 100,  # Critical temperature
        'x_rms_vel': 0.5,    # mm/s (ISO 10816 standard)
        'z_rms_vel': 0.5,    # mm/s
        'x_peak_vel': 0.8,   # mm/s
        'z_peak_vel': 0.8,   # mm/s
        'x_rms_accel': 4.0,  # m/s² 
        'z_rms_accel': 4.0,  # m/s²
        'x_peak_accel': 6.0, # m/s²
        'z_peak_accel': 6.0  # m/s²
    }
    
    # Only use thresholds for columns that exist in the dataframe
    thresholds_rul = {k: v for k, v in default_thresholds_rul.items() if k in df.columns}

# Add a sidebar for file upload
st.sidebar.title("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload XLSX file with sensor data:", type="xlsx")

data_load_state = st.sidebar.empty()

# Generate sample data by default
if df is None:
    df = generate_sample_data()
    data_load_state.info("Using sample data. Upload an Excel file to use your own data.")

# Load data when file is uploaded
if uploaded_file is not None:
    result = load_dataset(uploaded_file)
    if result.startswith("Error"):
        data_load_state.error(result)
    else:
        data_load_state.success(result)

# Main content layout
st.title("Industrial Equipment Monitoring & RUL Prediction")
st.markdown(f"### Current Date: {current_date.strftime('%B %d, %Y')}")
st.markdown(f"#### Using {'uploaded data' if data_source == 'Uploaded' else 'sample data with 180 days of machine history'}")

# Time Series Overview
st.header("Time Series Analysis")

with st.expander("Sensor Data Over Time", expanded=True):
    time_series_tab1, time_series_tab2, time_series_tab3 = st.tabs(["Velocity Parameters", "Acceleration Parameters", "Temperature"])
    
    with time_series_tab1:
        # Create time series plot for velocity parameters
        fig_vel = go.Figure()
        
        # Add RMS velocity data
        fig_vel.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['x_rms_vel'],
            mode='lines',
            name='X-RMS Velocity',
            line=dict(color='blue')
        ))
        
        fig_vel.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['z_rms_vel'],
            mode='lines',
            name='Z-RMS Velocity',
            line=dict(color='green')
        ))
        
        # Add peak velocity data
        fig_vel.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['x_peak_vel'],
            mode='lines',
            name='X-Peak Velocity',
            line=dict(color='blue', dash='dash')
        ))
        
        fig_vel.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['z_peak_vel'],
            mode='lines',
            name='Z-Peak Velocity',
            line=dict(color='green', dash='dash')
        ))
        
        # Add threshold lines
        fig_vel.add_shape(
            type="line",
            x0=df['timestamp'].min(),
            y0=thresholds['x_rms_vel'],
            x1=df['timestamp'].max(),
            y1=thresholds['x_rms_vel'],
            line=dict(color="blue", width=1, dash="dot"),
        )
        
        fig_vel.add_shape(
            type="line",
            x0=df['timestamp'].min(),
            y0=thresholds['z_rms_vel'],
            x1=df['timestamp'].max(),
            y1=thresholds['z_rms_vel'],
            line=dict(color="green", width=1, dash="dot"),
        )
        
        fig_vel.update_layout(
            title="Machine Velocity Parameters Over Time",
            xaxis_title="Date",
            yaxis_title="Velocity (mm/s)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_vel, use_container_width=True)
        
    with time_series_tab2:
        # Create time series plot for acceleration parameters
        fig_accel = go.Figure()
        
        # Add RMS acceleration data
        fig_accel.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['x_rms_accel'],
            mode='lines',
            name='X-RMS Acceleration',
            line=dict(color='red')
        ))
        
        fig_accel.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['z_rms_accel'],
            mode='lines',
            name='Z-RMS Acceleration',
            line=dict(color='purple')
        ))
        
        # Add peak acceleration data
        fig_accel.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['x_peak_accel'],
            mode='lines',
            name='X-Peak Acceleration',
            line=dict(color='red', dash='dash')
        ))
        
        fig_accel.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['z_peak_accel'],
            mode='lines',
            name='Z-Peak Acceleration',
            line=dict(color='purple', dash='dash')
        ))
        
        # Add threshold lines
        fig_accel.add_shape(
            type="line",
            x0=df['timestamp'].min(),
            y0=thresholds['x_rms_accel'],
            x1=df['timestamp'].max(),
            y1=thresholds['x_rms_accel'],
            line=dict(color="red", width=1, dash="dot"),
        )
        
        fig_accel.add_shape(
            type="line",
            x0=df['timestamp'].min(),
            y0=thresholds['z_rms_accel'],
            x1=df['timestamp'].max(),
            y1=thresholds['z_rms_accel'],
            line=dict(color="purple", width=1, dash="dot"),
        )
        
        fig_accel.update_layout(
            title="Machine Acceleration Parameters Over Time",
            xaxis_title="Date",
            yaxis_title="Acceleration (m/s²)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_accel, use_container_width=True)
    
    with time_series_tab3:
        # Create time series plot for temperature
        fig_temp = go.Figure()
        
        fig_temp.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['temperature'],
            mode='lines',
            name='Temperature',
            line=dict(color='orange')
        ))
        
        # Add threshold line
        fig_temp.add_shape(
            type="line",
            x0=df['timestamp'].min(),
            y0=thresholds['temperature'],
            x1=df['timestamp'].max(),
            y1=thresholds['temperature'],
            line=dict(color="red", width=1, dash="dot"),
        )
        
        fig_temp.update_layout(
            title="Machine Temperature Over Time",
            xaxis_title="Date",
            yaxis_title="Temperature",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)
        
    # Add an extra explanation
    st.markdown("""
    **Note:** In these time series charts, you can observe:
    - Gradually increasing trend, especially after day 120
    - Weekly cyclical patterns visible in the sensor data  
    - Threshold lines showing 95th percentile values for each parameter
    """)

# RUL Charts (Plotly Line Charts with time on X-axis)
st.header("Remaining Useful Life (RUL) Analysis")

with st.expander("RUL Over Time", expanded=True):
    # Create a time-based RUL chart
    fig_rul = go.Figure()
    
    # Add RUL traces for parameters
    fig_rul.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature_rul'],
        mode='lines',
        name='Temperature RUL',
        line=dict(color='orange')
    ))
    
    fig_rul.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['x_rms_vel_rul'],
        mode='lines',
        name='X-RMS Velocity RUL',
        line=dict(color='blue')
    ))
    
    fig_rul.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['z_rms_vel_rul'],
        mode='lines',
        name='Z-RMS Velocity RUL',
        line=dict(color='green')
    ))
    
    fig_rul.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['x_rms_accel_rul'],
        mode='lines',
        name='X-RMS Acceleration RUL',
        line=dict(color='red')
    ))
    
    fig_rul.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['z_rms_accel_rul'],
        mode='lines',
        name='Z-RMS Acceleration RUL',
        line=dict(color='purple')
    ))
    
    # Add overall health index
    fig_rul.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['health_index'],
        mode='lines',
        name='Overall Health Index',
        line=dict(color='black', width=3)
    ))
    
    # Add maintenance threshold line
    fig_rul.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=maintenance_threshold,
        x1=df['timestamp'].max(),
        y1=maintenance_threshold,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add annotation for maintenance threshold
    fig_rul.add_annotation(
        x=df['timestamp'].max(),
        y=maintenance_threshold,
        text=f"Maintenance Threshold ({maintenance_threshold}%)",
        showarrow=False,
        yshift=10
    )
    
    fig_rul.update_layout(
        title="Remaining Useful Life (RUL) Over Time",
        xaxis_title="Date",
        yaxis_title="RUL (%)",
        yaxis_range=[0, 100],
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_rul, use_container_width=True)
    
    # Calculate the latest RUL values
    latest_date = df['timestamp'].max()
    latest_health = df['health_index'].iloc[-1]
    latest_temp_rul = df['temperature_rul'].iloc[-1]
    latest_x_vel_rul = df['x_rms_vel_rul'].iloc[-1]
    latest_z_vel_rul = df['z_rms_vel_rul'].iloc[-1]
    latest_x_accel_rul = df['x_rms_accel_rul'].iloc[-1]
    latest_z_accel_rul = df['z_rms_accel_rul'].iloc[-1]
    
    st.markdown(f"""
    ### Latest RUL Values (as of {latest_date.strftime('%B %d, %Y')})
    
    - **Overall Health Index**: {latest_health:.1f}%
    - **Temperature RUL**: {latest_temp_rul:.1f}%
    - **X-RMS Velocity RUL**: {latest_x_vel_rul:.1f}%
    - **Z-RMS Velocity RUL**: {latest_z_vel_rul:.1f}%
    - **X-RMS Acceleration RUL**: {latest_x_accel_rul:.1f}%
    - **Z-RMS Acceleration RUL**: {latest_z_accel_rul:.1f}%
    """)

# RUL Forecast and Maintenance Scheduling
st.header("RUL Forecast & Maintenance Schedule")

with st.expander("Predictive Maintenance Analysis", expanded=True):
    # Calculate degradation rates based on historical data
    first_health = df['health_index'].iloc[0]
    last_health = df['health_index'].iloc[-1]
    time_diff_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / (60 * 60 * 24)
    health_degradation = (first_health - last_health) / time_diff_days
    
    # Calculate days until maintenance threshold
    days_to_maintenance = (last_health - maintenance_threshold) / health_degradation if health_degradation > 0 else 999
    days_to_maintenance = max(0, days_to_maintenance)
    
    # Calculate days until failure (health = 0)
    days_to_failure = last_health / health_degradation if health_degradation > 0 else 999
    
    # Calculate forecast dates
    maintenance_date = current_date + timedelta(days=days_to_maintenance)
    failure_date = current_date + timedelta(days=days_to_failure)
    
    # Create forecast dataframe for next 90 days
    future_days = 90
    forecast_dates = pd.date_range(start=current_date, periods=future_days+1, freq='D')
    
    forecast_df = pd.DataFrame({'Date': forecast_dates})
    forecast_df['Projected_Health'] = [max(0, last_health - (health_degradation * day)) for day in range(future_days+1)]
    
    # Create the forecast plot
    fig_forecast = go.Figure()
    
    # Add historical health data
    fig_forecast.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['health_index'],
        mode='lines',
        name='Historical Health',
        line=dict(color='blue')
    ))
    
    # Add forecasted health data
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Projected_Health'],
        mode='lines',
        name='Projected Health',
        line=dict(color='red')
    ))
    
    # Add threshold line
    fig_forecast.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=maintenance_threshold,
        x1=forecast_df['Date'].max(),
        y1=maintenance_threshold,
        line=dict(color="orange", width=2, dash="dash"),
    )
    
    # Add vertical line for current date
    fig_forecast.add_shape(
        type="line",
        x0=current_date,
        y0=0,
        x1=current_date,
        y1=100,
        line=dict(color="green", width=2),
    )
    
    # Add annotation for current date
    fig_forecast.add_annotation(
        x=current_date,
        y=95,
        text="Today",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
    )
    
    # Add annotation for maintenance date if within view
    if maintenance_date <= forecast_df['Date'].max():
        fig_forecast.add_shape(
            type="line", 
            x0=maintenance_date,
            y0=0,
            x1=maintenance_date,
            y1=maintenance_threshold,
            line=dict(color="orange", width=2),
        )
        
        fig_forecast.add_annotation(
            x=maintenance_date,
            y=maintenance_threshold,
            text=f"Maintenance: {maintenance_date.strftime('%b %d')}",
            showarrow=True,
            arrowhead=1,
        )
    
    fig_forecast.update_layout(
        title="Equipment Health Forecast",
        xaxis_title="Date",
        yaxis_title="Health Index (%)",
        yaxis_range=[0, 100],
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Create a maintenance schedule display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Maintenance Schedule")
        
        maintenance_status = ""
        if days_to_maintenance <= 7:
            maintenance_status = "⚠️ URGENT: Schedule immediate maintenance!"
        elif days_to_maintenance <= 30:
            maintenance_status = "⚠️ WARNING: Schedule maintenance soon"
        else:
            maintenance_status = "✅ GOOD: No immediate maintenance required"
        
        st.markdown(f"""
        **Current Health**: {last_health:.1f}%
        
        **Degradation Rate**: {health_degradation:.2f}% per day
        
        **Maintenance Required In**: {days_to_maintenance:.1f} days
        
        **Projected Maintenance Date**: {maintenance_date.strftime('%B %d, %Y')}
        
        **Status**: {maintenance_status}
        """)
    
    with col2:
        st.markdown("### Health Analysis")
        
        # Create a gauge chart for current health
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=last_health,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current Health (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "red"},
                    {'range': [25, 50], 'color': "orange"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': maintenance_threshold
                }
            }
        ))
        
        st.plotly_chart(fig_gauge, use_container_width=True)

# Maintenance Optimization
st.header("Maintenance Optimization")

with st.expander("Cost Optimization Analysis", expanded=True):
    # Default values for cost analysis
    machine_id = "Motor-01"
    machine_cost = 10000
    downtime_cost = 2000
    maintenance_cost = 500
    maintenance_days = 1
    
    # Calculate costs
    cost_of_maintenance = maintenance_cost + (maintenance_days * downtime_cost)
    cost_of_failure = machine_cost + (5 * downtime_cost)  # Assume 5 days downtime for replacement
    
    # Calculate optimal maintenance date (80% of time to failure)
    optimal_maintenance_day = min(days_to_maintenance, days_to_failure * 0.8)
    optimal_maintenance_date = current_date + timedelta(days=optimal_maintenance_day)
    
    # Create visualization of maintenance optimization
    future_span = int(max(days_to_failure * 1.2, 90))  # Look ahead for 120% of failure time or 90 days
    timeline_df = pd.DataFrame({
        'Day': range(0, future_span),
        'Health': [max(0, last_health - (health_degradation * d)) for d in range(0, future_span)],
    })
    
    # Add maintenance threshold
    timeline_df['Maintenance_Threshold'] = maintenance_threshold
    
    fig_optimize = go.Figure()
    
    # Plot health over time
    fig_optimize.add_trace(go.Scatter(
        x=timeline_df['Day'],
        y=timeline_df['Health'],
        mode='lines',
        name='Equipment Health'
    ))
    
    # Add maintenance threshold line
    fig_optimize.add_shape(
        type="line",
        x0=0,
        y0=maintenance_threshold,
        x1=timeline_df['Day'].max(),
        y1=maintenance_threshold,
        line=dict(color="orange", width=2, dash="dash"),
    )
    
    # Add annotation for optimal maintenance
    fig_optimize.add_shape(
        type="line",
        x0=optimal_maintenance_day,
        y0=0,
        x1=optimal_maintenance_day,
        y1=timeline_df['Health'].iloc[int(optimal_maintenance_day)] if int(optimal_maintenance_day) < len(timeline_df) else 0,
        line=dict(color="green", width=2),
    )
    
    fig_optimize.add_annotation(
        x=optimal_maintenance_day,
        y=timeline_df['Health'].iloc[int(optimal_maintenance_day)] if int(optimal_maintenance_day) < len(timeline_df) else 0,
        text="Optimal Maintenance Point",
        showarrow=True,
        arrowhead=1,
    )
    
    # Add failure point annotation
    if days_to_failure < future_span:
        fig_optimize.add_shape(
            type="line",
            x0=days_to_failure,
            y0=0,
            x1=days_to_failure,
            y1=10,  # Just a little above zero
            line=dict(color="red", width=2),
        )
        
        fig_optimize.add_annotation(
            x=days_to_failure,
            y=5,
            text="Projected Failure Point",
            showarrow=True,
            arrowhead=1,
        )
    
    fig_optimize.update_layout(
        title="Equipment Health Projection & Optimal Maintenance Point",
        xaxis_title="Days from Now",
        yaxis_title="Health Index (%)",
        yaxis_range=[0, 100],
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_optimize, use_container_width=True)
    
    # Display maintenance recommendations
    st.markdown(f"""
    ### Maintenance Cost Optimization
    
    Based on current health ({last_health:.1f}%) and degradation rate ({health_degradation:.2f}% per day):
    
    - **Maintenance threshold crossed in**: {days_to_maintenance:.1f} days
    - **Projected failure in**: {days_to_failure:.1f} days
    - **Optimal maintenance point**: {optimal_maintenance_day:.1f} days from now
    - **Recommended maintenance date**: {optimal_maintenance_date.strftime('%B %d, %Y')}
    """)

# Create a detailed health timeline plot
st.header("Detailed Health Timeline")

# Create timeline showing key maintenance events
timeline_days = 180  # Past 180 days + future 90 days
past_days = 180
future_days = 90

# Create a DataFrame with past and future dates
all_dates = pd.date_range(
    start=current_date - timedelta(days=past_days),
    end=current_date + timedelta(days=future_days),
    freq='D'
)

# Create the timeline DataFrame
timeline_df = pd.DataFrame({'Date': all_dates})
timeline_df['Day'] = [(d - current_date).days for d in timeline_df['Date']]
timeline_df['Position'] = range(len(timeline_df))

# Add events to the timeline
events = []

# Add degradation start event
degradation_start_date = current_date - timedelta(days=60)
events.append({
    'Date': degradation_start_date,
    'Day': (degradation_start_date - current_date).days,
    'Event': 'Degradation Acceleration',
    'Description': 'Machine started showing increased degradation rates',
    'Color': 'orange'
})

# Add current date
events.append({
    'Date': current_date,
    'Day': 0,
    'Event': 'Current Date',
    'Description': f'Today: {current_date.strftime("%B %d, %Y")}',
    'Color': 'green'
})

# Add maintenance date
events.append({
    'Date': maintenance_date,
    'Day': days_to_maintenance,
    'Event': 'Maintenance Due',
    'Description': f'Schedule maintenance by {maintenance_date.strftime("%B %d, %Y")}',
    'Color': 'red'
})

# Add failure date if it's within our time window
if days_to_failure < future_days:
    events.append({
        'Date': failure_date,
        'Day': days_to_failure,
        'Event': 'Projected Failure',
        'Description': f'Estimated failure on {failure_date.strftime("%B %d, %Y")} if no maintenance is performed',
        'Color': 'darkred'
    })

# Create the timeline visualization
fig_timeline = go.Figure()

# Add a line showing the health trend
health_values = []

# For past dates, use actual data from df if available
for date in timeline_df['Date']:
    if date <= current_date:
        # Try to find the exact date in df, otherwise use the closest one
        matching_df_rows = df[df['timestamp'] == date]
        if not matching_df_rows.empty:
            health_values.append(float(matching_df_rows['health_index'].iloc[0]))
        else:
            # Find closest date
            closest_date = df['timestamp'].iloc[(df['timestamp'] - date).abs().argsort()[0]]
            health_values.append(float(df[df['timestamp'] == closest_date]['health_index'].iloc[0]))
    else:
        # For future dates, use the projected values
        days_from_now = (date - current_date).days
        projected_health = max(0, last_health - (health_degradation * days_from_now))
        health_values.append(projected_health)

timeline_df['Health'] = health_values

# Add the health line
fig_timeline.add_trace(go.Scatter(
    x=timeline_df['Date'],
    y=timeline_df['Health'],
    mode='lines',
    name='Health Index',
    line=dict(color='blue', width=2)
))

# Add threshold line
fig_timeline.add_shape(
    type="line",
    x0=timeline_df['Date'].min(),
    y0=maintenance_threshold,
    x1=timeline_df['Date'].max(),
    y1=maintenance_threshold,
    line=dict(color="red", width=2, dash="dash"),
)

# Add vertical lines and annotations for events
for event in events:
    # Add vertical line for each event
    fig_timeline.add_shape(
        type="line",
        x0=event['Date'],
        y0=0,
        x1=event['Date'],
        y1=100,
        line=dict(color=event['Color'], width=2),
    )
    
    # Add annotation
    fig_timeline.add_annotation(
        x=event['Date'],
        y=90,  # Position at the top
        text=event['Event'],
        showarrow=True,
        arrowhead=1,
        arrowcolor=event['Color'],
        bgcolor="white",
        opacity=0.8
    )

# Add current date vertical line
fig_timeline.add_shape(
    type="line",
    x0=current_date,
    y0=0,
    x1=current_date,
    y1=100,
    line=dict(color="green", width=3),
)

# Update layout
fig_timeline.update_layout(
    title="Machine Health Timeline with Key Events",
    xaxis_title="Date",
    yaxis_title="Health Index (%)",
    yaxis_range=[0, 100],
    hovermode="x unified"
)

st.plotly_chart(fig_timeline, use_container_width=True)

# Show events table with descriptions
events_df = pd.DataFrame(events)[['Date', 'Event', 'Description']]
events_df['Date'] = events_df['Date'].dt.strftime('%B %d, %Y')
st.table(events_df)