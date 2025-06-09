"""
Streamlit Brainwave Visualization App
Real-time EEG brainwave analysis dashboard showing Alpha, Beta, Theta, Delta, and Gamma waves
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
import threading
import queue

# Import our EEG modules
from eeg_collector import EEGDataCollector
from brainwave_analyzer import BrainwaveAnalyzer

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 Brainwave Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .brainwave-delta { color: #ff7f0e; }
    .brainwave-theta { color: #2ca02c; }
    .brainwave-alpha { color: #d62728; }
    .brainwave-beta { color: #9467bd; }
    .brainwave-gamma { color: #8c564b; }
</style>
""", unsafe_allow_html=True)

class StreamlitBrainwaveApp:
    def __init__(self):
        self.collector = None
        self.analyzer = BrainwaveAnalyzer(sample_rate=256)
        self.is_running = False
        self.data_queue = queue.Queue()
        self.collection_thread = None
        
        # Data storage for visualization
        self.time_series_data = {
            'timestamps': [],
            'delta': [],
            'theta': [],
            'alpha': [],
            'beta': [],
            'gamma': []
        }
        
        self.mental_states_data = {
            'timestamps': [],
            'relaxation': [],
            'concentration': [],
            'meditation': [],
            'alertness': [],
            'drowsiness': []
        }
        
        # Keep only last N points for performance
        self.max_points = 100
    
    def start_data_collection(self):
        """Start EEG data collection in background thread"""
        if not self.is_running:
            self.collector = EEGDataCollector(headset_type="simulation", sample_rate=256)
            self.collector.start_collection()
            self.is_running = True
            
            # Start background thread for data collection
            self.collection_thread = threading.Thread(target=self._collect_data_background)
            self.collection_thread.daemon = True
            self.collection_thread.start()
    
    def stop_data_collection(self):
        """Stop EEG data collection"""
        if self.is_running:
            self.is_running = False
            if self.collector:
                self.collector.cleanup()
            if self.collection_thread:
                self.collection_thread.join(timeout=1)
    
    def _collect_data_background(self):
        """Background thread function for continuous data collection"""
        while self.is_running:
            try:
                # Get latest data
                data, timestamps = self.collector.get_latest_data(512)  # 2 seconds of data
                
                if len(data) > 256:  # Need sufficient data for analysis
                    # Analyze the data
                    band_powers = self.analyzer.extract_band_power(np.array(data))
                    mental_states = self.analyzer.detect_mental_state(np.array(data))
                    
                    # Put data in queue for main thread
                    self.data_queue.put({
                        'timestamp': datetime.now(),
                        'band_powers': band_powers,
                        'mental_states': mental_states,
                        'raw_data': data[-256:]  # Last 1 second for raw display
                    })
                
                time.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                st.error(f"Data collection error: {e}")
                break
    
    def update_data_storage(self):
        """Update data storage with latest values from queue"""
        updated = False
        
        while not self.data_queue.empty():
            try:
                data_point = self.data_queue.get_nowait()
                
                # Add timestamp
                timestamp = data_point['timestamp']
                
                # Update band powers
                self.time_series_data['timestamps'].append(timestamp)
                for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    if band in data_point['band_powers']:
                        power = np.mean(data_point['band_powers'][band])
                        self.time_series_data[band].append(power)
                    else:
                        self.time_series_data[band].append(0)
                
                # Update mental states
                self.mental_states_data['timestamps'].append(timestamp)
                for state in ['relaxation', 'concentration', 'meditation', 'alertness', 'drowsiness']:
                    if state in data_point['mental_states']:
                        self.mental_states_data[state].append(data_point['mental_states'][state])
                    else:
                        self.mental_states_data[state].append(0)
                
                updated = True
                
            except queue.Empty:
                break
        
        # Keep only recent data points
        if len(self.time_series_data['timestamps']) > self.max_points:
            for key in self.time_series_data:
                self.time_series_data[key] = self.time_series_data[key][-self.max_points:]
            
            for key in self.mental_states_data:
                self.mental_states_data[key] = self.mental_states_data[key][-self.max_points:]
        
        return updated
    
    def create_brainwave_chart(self):
        """Create real-time brainwave frequency bands chart"""
        if not self.time_series_data['timestamps']:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)
        
        fig = go.Figure()
        
        colors = {
            'delta': '#ff7f0e',
            'theta': '#2ca02c', 
            'alpha': '#d62728',
            'beta': '#9467bd',
            'gamma': '#8c564b'
        }
        
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            fig.add_trace(go.Scatter(
                x=self.time_series_data['timestamps'],
                y=self.time_series_data[band],
                mode='lines+markers',
                name=f'{band.capitalize()} ({self.get_frequency_range(band)})',
                line=dict(color=colors[band], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="🌊 Real-time Brainwave Frequency Bands",
            xaxis_title="Time",
            yaxis_title="Power (µV²)",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_mental_states_chart(self):
        """Create mental states visualization"""
        if not self.mental_states_data['timestamps']:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)
        
        fig = go.Figure()
        
        colors = {
            'relaxation': '#2ca02c',
            'concentration': '#d62728',
            'meditation': '#9467bd',
            'alertness': '#ff7f0e',
            'drowsiness': '#8c564b'
        }
        
        for state in ['relaxation', 'concentration', 'meditation', 'alertness', 'drowsiness']:
            fig.add_trace(go.Scatter(
                x=self.mental_states_data['timestamps'],
                y=self.mental_states_data[state],
                mode='lines+markers',
                name=state.capitalize(),
                line=dict(color=colors[state], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="🧠 Mental State Indicators",
            xaxis_title="Time",
            yaxis_title="State Index",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_current_state_gauge(self):
        """Create gauge showing current dominant mental state"""
        if not self.mental_states_data['timestamps']:
            return go.Figure()
        
        # Get latest mental states
        latest_states = {}
        for state in ['relaxation', 'concentration', 'meditation', 'alertness', 'drowsiness']:
            if self.mental_states_data[state]:
                latest_states[state] = self.mental_states_data[state][-1]
        
        if not latest_states:
            return go.Figure()
        
        # Find dominant state
        dominant_state = max(latest_states.items(), key=lambda x: x[1])
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = dominant_state[1],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Current State: {dominant_state[0].capitalize()}"},
            delta = {'reference': 1.0},
            gauge = {
                'axis': {'range': [None, 3]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 2], 'color': "gray"},
                    {'range': [2, 3], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_band_power_bars(self):
        """Create current band power bar chart"""
        if not self.time_series_data['timestamps']:
            return go.Figure()
        
        # Get latest band powers
        latest_powers = {}
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if self.time_series_data[band]:
                latest_powers[band] = self.time_series_data[band][-1]
        
        if not latest_powers:
            return go.Figure()
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(latest_powers.keys()),
                y=list(latest_powers.values()),
                marker_color=colors,
                text=[f'{v:.2f}' for v in latest_powers.values()],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Current Brainwave Band Powers",
            xaxis_title="Frequency Band",
            yaxis_title="Power (µV²)",
            height=300
        )
        
        return fig
    
    def get_frequency_range(self, band):
        """Get frequency range for a band"""
        ranges = {
            'delta': '0.5-4 Hz',
            'theta': '4-8 Hz',
            'alpha': '8-13 Hz',
            'beta': '13-30 Hz',
            'gamma': '30-100 Hz'
        }
        return ranges.get(band, '')

def main():
    # Initialize the app
    if 'app' not in st.session_state:
        st.session_state.app = StreamlitBrainwaveApp()
    
    app = st.session_state.app
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Real-time Brainwave Monitor</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start", key="start_btn"):
            app.start_data_collection()
            st.sidebar.success("Started data collection!")
    
    with col2:
        if st.button("⏹️ Stop", key="stop_btn"):
            app.stop_data_collection()
            st.sidebar.info("Stopped data collection")
    
    # Status indicator
    status = "🟢 Running" if app.is_running else "🔴 Stopped"
    st.sidebar.markdown(f"**Status:** {status}")
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (1s)", value=True)
    show_raw_data = st.sidebar.checkbox("Show raw EEG signal", value=False)
    
    # Information panel
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Brainwave Bands")
    st.sidebar.markdown("""
    - **Delta (0.5-4 Hz)**: Deep sleep, unconscious
    - **Theta (4-8 Hz)**: Meditation, creativity
    - **Alpha (8-13 Hz)**: Relaxation, calm focus
    - **Beta (13-30 Hz)**: Active thinking, concentration
    - **Gamma (30-100 Hz)**: High-level cognitive processing
    """)
    
    # Main content area
    if auto_refresh and app.is_running:
        # Auto-refresh every second
        time.sleep(1)
        st.rerun()
    
    # Update data
    app.update_data_storage()
    
    # Main dashboard
    if app.is_running or app.time_series_data['timestamps']:
        # Current state metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        if app.time_series_data['timestamps']:
            with col1:
                delta_val = app.time_series_data['delta'][-1] if app.time_series_data['delta'] else 0
                st.metric("Delta", f"{delta_val:.2f} µV²", delta=None)
            
            with col2:
                theta_val = app.time_series_data['theta'][-1] if app.time_series_data['theta'] else 0
                st.metric("Theta", f"{theta_val:.2f} µV²", delta=None)
            
            with col3:
                alpha_val = app.time_series_data['alpha'][-1] if app.time_series_data['alpha'] else 0
                st.metric("Alpha", f"{alpha_val:.2f} µV²", delta=None)
            
            with col4:
                beta_val = app.time_series_data['beta'][-1] if app.time_series_data['beta'] else 0
                st.metric("Beta", f"{beta_val:.2f} µV²", delta=None)
            
            with col5:
                gamma_val = app.time_series_data['gamma'][-1] if app.time_series_data['gamma'] else 0
                st.metric("Gamma", f"{gamma_val:.2f} µV²", delta=None)
        
        # Charts row 1
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(app.create_brainwave_chart(), use_container_width=True)
        
        with col2:
            st.plotly_chart(app.create_current_state_gauge(), use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(app.create_mental_states_chart(), use_container_width=True)
        
        with col2:
            st.plotly_chart(app.create_band_power_bars(), use_container_width=True)
        
        # Data summary
        if app.time_series_data['timestamps']:
            st.markdown("---")
            st.markdown("### 📈 Current Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Brainwave Activity:**")
                latest_bands = {
                    'Delta': app.time_series_data['delta'][-1] if app.time_series_data['delta'] else 0,
                    'Theta': app.time_series_data['theta'][-1] if app.time_series_data['theta'] else 0,
                    'Alpha': app.time_series_data['alpha'][-1] if app.time_series_data['alpha'] else 0,
                    'Beta': app.time_series_data['beta'][-1] if app.time_series_data['beta'] else 0,
                    'Gamma': app.time_series_data['gamma'][-1] if app.time_series_data['gamma'] else 0
                }
                
                dominant_band = max(latest_bands.items(), key=lambda x: x[1])
                st.write(f"🏆 **Dominant Band:** {dominant_band[0]} ({dominant_band[1]:.2f} µV²)")
                
                # Create a simple bar chart of current values
                df_bands = pd.DataFrame(list(latest_bands.items()), columns=['Band', 'Power'])
                st.bar_chart(df_bands.set_index('Band'))
            
            with col2:
                if app.mental_states_data['timestamps']:
                    st.markdown("**Mental States:**")
                    latest_states = {
                        'Relaxation': app.mental_states_data['relaxation'][-1] if app.mental_states_data['relaxation'] else 0,
                        'Concentration': app.mental_states_data['concentration'][-1] if app.mental_states_data['concentration'] else 0,
                        'Meditation': app.mental_states_data['meditation'][-1] if app.mental_states_data['meditation'] else 0,
                        'Alertness': app.mental_states_data['alertness'][-1] if app.mental_states_data['alertness'] else 0,
                        'Drowsiness': app.mental_states_data['drowsiness'][-1] if app.mental_states_data['drowsiness'] else 0
                    }
                    
                    dominant_state = max(latest_states.items(), key=lambda x: x[1])
                    st.write(f"🧠 **Dominant State:** {dominant_state[0]} ({dominant_state[1]:.2f})")
                    
                    # Create a simple bar chart of current states
                    df_states = pd.DataFrame(list(latest_states.items()), columns=['State', 'Value'])
                    st.bar_chart(df_states.set_index('State'))
        
        # Raw data display (optional)
        if show_raw_data and app.time_series_data['timestamps']:
            st.markdown("---")
            st.markdown("### 📊 Raw Data Tables")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Recent Brainwave Data:**")
                if len(app.time_series_data['timestamps']) > 0:
                    recent_data = pd.DataFrame({
                        'Time': app.time_series_data['timestamps'][-10:],
                        'Delta': app.time_series_data['delta'][-10:],
                        'Theta': app.time_series_data['theta'][-10:],
                        'Alpha': app.time_series_data['alpha'][-10:],
                        'Beta': app.time_series_data['beta'][-10:],
                        'Gamma': app.time_series_data['gamma'][-10:]
                    })
                    st.dataframe(recent_data, use_container_width=True)
            
            with col2:
                st.markdown("**Recent Mental States:**")
                if len(app.mental_states_data['timestamps']) > 0:
                    recent_states = pd.DataFrame({
                        'Time': app.mental_states_data['timestamps'][-10:],
                        'Relaxation': app.mental_states_data['relaxation'][-10:],
                        'Concentration': app.mental_states_data['concentration'][-10:],
                        'Meditation': app.mental_states_data['meditation'][-10:],
                        'Alertness': app.mental_states_data['alertness'][-10:],
                        'Drowsiness': app.mental_states_data['drowsiness'][-10:]
                    })
                    st.dataframe(recent_states, use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ### Welcome to the Brainwave Monitor! 🧠
        
        This app provides real-time visualization of EEG brainwave data including:
        
        - **🌊 Frequency Bands**: Delta, Theta, Alpha, Beta, Gamma waves
        - **🧠 Mental States**: Relaxation, Concentration, Meditation, Alertness, Drowsiness
        - **📊 Real-time Charts**: Live updating visualizations
        - **📈 Current Metrics**: Instant feedback on brain activity
        
        **To get started:**
        1. Click the "▶️ Start" button in the sidebar
        2. Watch the real-time brainwave data appear
        3. Analyze your mental states and brainwave patterns
        
        **Note:** This demo uses simulated EEG data. In a real application, this would connect to an actual EEG headset.
        """)
        
        # Show sample/demo charts with fake data
        st.markdown("### 📊 Sample Visualizations")
        
        # Create sample data for demonstration
        sample_time = pd.date_range(start=datetime.now() - timedelta(minutes=5), 
                                   end=datetime.now(), freq='1S')
        sample_data = {
            'Delta': np.random.normal(1, 0.3, len(sample_time)),
            'Theta': np.random.normal(8, 2, len(sample_time)),
            'Alpha': np.random.normal(15, 4, len(sample_time)),
            'Beta': np.random.normal(10, 3, len(sample_time)),
            'Gamma': np.random.normal(2, 0.5, len(sample_time))
        }
        
        # Sample brainwave chart
        fig_sample = go.Figure()
        colors = {'Delta': '#ff7f0e', 'Theta': '#2ca02c', 'Alpha': '#d62728', 
                 'Beta': '#9467bd', 'Gamma': '#8c564b'}
        
        for band, color in colors.items():
            fig_sample.add_trace(go.Scatter(
                x=sample_time,
                y=sample_data[band],
                mode='lines',
                name=band,
                line=dict(color=color, width=2)
            ))
        
        fig_sample.update_layout(
            title="Sample Brainwave Patterns",
            xaxis_title="Time",
            yaxis_title="Power (µV²)",
            height=400
        )
        
        st.plotly_chart(fig_sample, use_container_width=True)

if __name__ == "__main__":
    main()
