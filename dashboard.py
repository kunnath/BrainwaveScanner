"""
Real-time EEG Dashboard using Streamlit
Provides live visualization and analysis of brainwave data
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import threading
import queue
from datetime import datetime, timedelta
import json

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from eeg_collector import EEGDataCollector
    from brainwave_analyzer import BrainwaveAnalyzer
except ImportError:
    st.error("Required modules not found. Please ensure eeg_collector.py and brainwave_analyzer.py are in the same directory.")
    st.stop()

class RealTimeDashboard:
    """Real-time EEG monitoring dashboard"""
    
    def __init__(self):
        self.collector = None
        self.analyzer = BrainwaveAnalyzer(sample_rate=256)
        self.data_buffer = queue.Queue(maxsize=1000)
        self.is_monitoring = False
        
    def initialize_collector(self, headset_type: str):
        """Initialize EEG collector with specified headset"""
        if self.collector:
            self.collector.cleanup()
        self.collector = EEGDataCollector(headset_type=headset_type, sample_rate=256)
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.collector and not self.is_monitoring:
            self.collector.start_collection()
            self.is_monitoring = True
            
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.collector and self.is_monitoring:
            self.collector.stop_collection()
            self.is_monitoring = False
            
    def get_latest_analysis(self, window_size: int = 512):
        """Get latest data and analysis"""
        if not self.collector:
            return None, None, None
            
        data, timestamps = self.collector.get_latest_data(window_size)
        
        if len(data) < 100:  # Not enough data
            return None, None, None
            
        # Perform analysis
        try:
            band_powers = self.analyzer.extract_band_power(data)
            mental_states = self.analyzer.detect_mental_state(data)
            features = self.analyzer.extract_features(data)
            
            return data, band_powers, mental_states
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return None, None, None

def main():
    st.set_page_config(
        page_title="🧠 EEG Brainwave Monitor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Real-Time EEG Brainwave Monitor")
    st.markdown("---")
    
    # Initialize session state
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = RealTimeDashboard()
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'data_history' not in st.session_state:
        st.session_state.data_history = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
        
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    st.sidebar.header("🎧 EEG Setup")
    
    # Headset selection
    headset_options = {
        "Simulation (Demo)": "simulation",
        "Muse Headset": "muse", 
        "OpenBCI": "openbci",
        "NeuroSky": "neurosky",
        "Emotiv": "emotiv"
    }
    
    selected_headset = st.sidebar.selectbox(
        "Select EEG Headset:",
        list(headset_options.keys()),
        index=0
    )
    
    headset_type = headset_options[selected_headset]
    
    # Initialize collector button
    if st.sidebar.button("🔌 Initialize Headset"):
        with st.spinner("Initializing headset..."):
            dashboard.initialize_collector(headset_type)
            st.sidebar.success(f"✅ {selected_headset} initialized!")
    
    # Monitoring controls
    st.sidebar.header("📊 Monitoring")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.monitoring):
            dashboard.start_monitoring()
            st.session_state.monitoring = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.monitoring):
            dashboard.stop_monitoring()
            st.session_state.monitoring = False
            st.rerun()
    
    # Status indicator
    if st.session_state.monitoring:
        st.sidebar.success("🟢 Monitoring Active")
    else:
        st.sidebar.info("🔴 Monitoring Stopped")
    
    # Analysis settings
    st.sidebar.header("⚙️ Settings")
    
    update_rate = st.sidebar.slider("Update Rate (seconds)", 0.5, 5.0, 1.0, 0.5)
    window_size = st.sidebar.slider("Analysis Window (samples)", 256, 2048, 512, 256)
    
    # Main dashboard
    if st.session_state.monitoring and dashboard.collector:
        
        # Create placeholder containers
        status_container = st.container()
        metrics_container = st.container()
        charts_container = st.container()
        
        # Auto-refresh loop
        placeholder = st.empty()
        
        with placeholder.container():
            # Get latest data and analysis
            data, band_powers, mental_states = dashboard.get_latest_analysis(window_size)
            
            if data is not None:
                # Update history
                current_time = datetime.now()
                st.session_state.data_history.append({
                    'timestamp': current_time,
                    'data': data,
                    'band_powers': band_powers,
                    'mental_states': mental_states
                })
                
                # Keep only last 100 records
                if len(st.session_state.data_history) > 100:
                    st.session_state.data_history = st.session_state.data_history[-100:]
                
                # Status metrics
                with status_container:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📡 Samples", len(data))
                    with col2:
                        st.metric("📊 Channels", data.shape[1] if len(data.shape) > 1 else 1)
                    with col3:
                        st.metric("🕐 Duration", f"{len(data)/256:.1f}s")
                    with col4:
                        st.metric("📈 Sample Rate", "256 Hz")
                
                # Mental state metrics
                with metrics_container:
                    st.subheader("🧠 Mental State Indicators")
                    
                    cols = st.columns(len(mental_states))
                    for i, (state, value) in enumerate(mental_states.items()):
                        with cols[i]:
                            # Color coding based on value
                            if value > 0.7:
                                delta_color = "normal"
                            elif value > 0.4:
                                delta_color = "off"
                            else:
                                delta_color = "inverse"
                                
                            st.metric(
                                state.title(),
                                f"{value:.2f}",
                                delta=f"{value:.2f}",
                                delta_color=delta_color
                            )
                
                # Charts
                with charts_container:
                    
                    # Real-time signal plot
                    st.subheader("📈 Real-Time EEG Signal")
                    
                    fig_signal = go.Figure()
                    
                    if len(data.shape) > 1:
                        for ch in range(min(4, data.shape[1])):  # Show up to 4 channels
                            time_axis = np.arange(len(data)) / 256  # Convert to seconds
                            fig_signal.add_trace(go.Scatter(
                                x=time_axis,
                                y=data[:, ch],
                                mode='lines',
                                name=f'Channel {ch+1}',
                                line=dict(width=1)
                            ))
                    else:
                        time_axis = np.arange(len(data)) / 256
                        fig_signal.add_trace(go.Scatter(
                            x=time_axis,
                            y=data,
                            mode='lines',
                            name='EEG Signal',
                            line=dict(width=1)
                        ))
                    
                    fig_signal.update_layout(
                        title="EEG Signal (Latest Window)",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude (µV)",
                        height=300,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_signal, use_container_width=True)
                    
                    # Band power visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🌊 Frequency Band Powers")
                        
                        # Create bar chart for band powers
                        bands = list(band_powers.keys())
                        powers = [np.mean(band_powers[band]) for band in bands]
                        
                        fig_bands = px.bar(
                            x=bands,
                            y=powers,
                            title="Average Band Powers",
                            labels={'x': 'Frequency Band', 'y': 'Power (µV²)'},
                            color=powers,
                            color_continuous_scale='viridis'
                        )
                        
                        fig_bands.update_layout(height=300)
                        st.plotly_chart(fig_bands, use_container_width=True)
                    
                    with col2:
                        st.subheader("🎯 Mental States")
                        
                        # Radar chart for mental states
                        states = list(mental_states.keys())
                        values = list(mental_states.values())
                        
                        fig_radar = go.Figure()
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=states,
                            fill='toself',
                            name='Mental States'
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=False,
                            height=300,
                            title="Mental State Profile"
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Historical trends
                    if len(st.session_state.data_history) > 5:
                        st.subheader("📊 Historical Trends")
                        
                        # Extract historical data
                        timestamps = [record['timestamp'] for record in st.session_state.data_history[-20:]]
                        
                        # Mental states over time
                        mental_trends = {}
                        for state in mental_states.keys():
                            mental_trends[state] = [record['mental_states'][state] 
                                                  for record in st.session_state.data_history[-20:]]
                        
                        fig_trends = go.Figure()
                        for state, values in mental_trends.items():
                            fig_trends.add_trace(go.Scatter(
                                x=timestamps,
                                y=values,
                                mode='lines+markers',
                                name=state.title(),
                                line=dict(width=2)
                            ))
                        
                        fig_trends.update_layout(
                            title="Mental States Over Time",
                            xaxis_title="Time",
                            yaxis_title="State Index",
                            height=300,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_trends, use_container_width=True)
                
                # Data export
                st.sidebar.header("💾 Data Export")
                
                if st.sidebar.button("📥 Export Current Session"):
                    # Prepare data for export
                    export_data = {
                        'session_info': {
                            'start_time': st.session_state.data_history[0]['timestamp'].isoformat() if st.session_state.data_history else None,
                            'end_time': current_time.isoformat(),
                            'headset_type': headset_type,
                            'sample_rate': 256,
                            'total_samples': len(st.session_state.data_history)
                        },
                        'data': []
                    }
                    
                    for record in st.session_state.data_history:
                        export_data['data'].append({
                            'timestamp': record['timestamp'].isoformat(),
                            'mental_states': record['mental_states'],
                            'band_powers': {k: v.tolist() if hasattr(v, 'tolist') else v 
                                          for k, v in record['band_powers'].items()}
                        })
                    
                    # Save to file
                    filename = f"eeg_session_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                    filepath = os.path.join("/Users/kunnath/Projects/brainwave", filename)
                    
                    with open(filepath, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                    
                    st.sidebar.success(f"✅ Data exported to {filename}")
                    
                    # Provide download link
                    with open(filepath, 'r') as f:
                        st.sidebar.download_button(
                            label="📱 Download Session Data",
                            data=f.read(),
                            file_name=filename,
                            mime="application/json"
                        )
            
            else:
                st.info("⏳ Waiting for EEG data...")
                st.markdown("Make sure your headset is properly connected and data is streaming.")
        
        # Auto-refresh
        time.sleep(update_rate)
        st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to the EEG Brainwave Monitor!
        
        This dashboard provides real-time visualization and analysis of your brainwave patterns.
        
        ### 🚀 Getting Started:
        
        1. **Select your EEG headset** from the sidebar
        2. **Click "Initialize Headset"** to set up the connection
        3. **Click "Start"** to begin monitoring
        4. **View real-time analysis** of your brainwaves
        
        ### 📊 Features:
        
        - **Real-time signal visualization**
        - **Frequency band analysis** (Delta, Theta, Alpha, Beta, Gamma)
        - **Mental state detection** (Relaxation, Concentration, Meditation, etc.)
        - **Historical trend tracking**
        - **Data export capabilities**
        
        ### 🎧 Supported Headsets:
        
        - **Muse 2/S** - Great for meditation and basic research
        - **OpenBCI** - Professional-grade research platform
        - **NeuroSky** - Affordable single-channel option
        - **Emotiv** - Advanced emotional state detection
        - **Simulation Mode** - For testing and demo purposes
        
        ### 💡 Tips:
        
        - Ensure proper electrode contact for best signal quality
        - Minimize movement and muscle tension during recording
        - Use in a quiet, distraction-free environment
        - Start with short sessions to get familiar with the interface
        """)
        
        # Demo section
        st.markdown("### 🎮 Demo Mode")
        st.info("Select 'Simulation (Demo)' to test the dashboard with simulated EEG data!")

if __name__ == "__main__":
    main()
