"""
Enhanced Brainwave Visualizer with Brain Image Overlay
Shows brainwave frequencies on a brain image for intuitive visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
from PIL import Image, ImageDraw, ImageFont
import cv2
import base64
from io import BytesIO

# Import our modules
from eeg_collector import EEGDataCollector
from brainwave_analyzer import BrainwaveAnalyzer

class BrainWaveVisualizer:
    """Enhanced brainwave visualizer with brain image overlay"""
    
    def __init__(self):
        self.collector = None
        self.analyzer = BrainwaveAnalyzer(sample_rate=256)
        self.is_collecting = False
        self.brain_regions = {
            'frontal': {'center': (180, 120), 'color': '#FF6B6B', 'activity': 'concentration'},
            'parietal': {'center': (180, 80), 'color': '#4ECDC4', 'activity': 'sensory_processing'},
            'temporal_left': {'center': (130, 140), 'color': '#45B7D1', 'activity': 'language'},
            'temporal_right': {'center': (230, 140), 'color': '#96CEB4', 'activity': 'music_spatial'},
            'occipital': {'center': (180, 180), 'color': '#FFEAA7', 'activity': 'visual'},
            'central': {'center': (180, 100), 'color': '#DDA0DD', 'activity': 'motor'}
        }
        
    def create_brain_image(self, width=360, height=300):
        """Create a simple brain outline image"""
        # Create a new image with transparent background
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw brain outline (simplified)
        # Main brain shape
        brain_outline = [
            (80, 150), (90, 120), (110, 100), (140, 85), (170, 80),
            (200, 80), (230, 85), (260, 100), (280, 120), (290, 150),
            (285, 180), (270, 200), (250, 215), (220, 225), (190, 230),
            (170, 230), (140, 225), (110, 215), (90, 200), (75, 180)
        ]
        
        # Draw brain outline
        draw.polygon(brain_outline, outline='#2C3E50', width=3, fill='#ECF0F1')
        
        # Draw central line (left-right hemisphere division)
        draw.line([(180, 80), (180, 230)], fill='#34495E', width=2)
        
        # Add region labels
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            
        # Label brain regions
        draw.text((140, 60), "FRONTAL", fill='#2C3E50', font=font, anchor="mm")
        draw.text((140, 190), "OCCIPITAL", fill='#2C3E50', font=font, anchor="mm")
        draw.text((100, 130), "TEMPORAL\nL", fill='#2C3E50', font=font, anchor="mm")
        draw.text((260, 130), "TEMPORAL\nR", fill='#2C3E50', font=font, anchor="mm")
        draw.text((180, 40), "PARIETAL", fill='#2C3E50', font=font, anchor="mm")
        
        return img
    
    def add_brainwave_overlay(self, brain_img, band_powers, mental_states):
        """Add brainwave activity overlay to brain image"""
        img = brain_img.copy()
        draw = ImageDraw.Draw(img)
        
        # Normalize band powers for visualization
        max_power = max([np.mean(power) for power in band_powers.values()]) + 1e-10
        
        # Map brainwaves to brain regions with activity intensity
        region_activities = {
            'frontal': band_powers.get('beta', [0])[0] / max_power,  # Beta for concentration
            'parietal': band_powers.get('alpha', [0])[0] / max_power,  # Alpha for relaxation
            'temporal_left': band_powers.get('theta', [0])[0] / max_power,  # Theta for creativity
            'temporal_right': band_powers.get('theta', [0])[0] / max_power,  # Theta for creativity
            'occipital': band_powers.get('alpha', [0])[0] / max_power,  # Alpha for visual processing
            'central': band_powers.get('gamma', [0])[0] / max_power,  # Gamma for motor activity
        }
        
        # Draw activity circles for each region
        for region, info in self.brain_regions.items():
            center = info['center']
            activity = region_activities.get(region, 0)
            
            # Calculate circle size based on activity (10-50 pixels radius)
            radius = int(10 + activity * 40)
            
            # Calculate alpha based on activity (20-200)
            alpha = int(20 + activity * 180)
            
            # Convert hex color to RGB
            color = info['color']
            if color.startswith('#'):
                color = color[1:]
            rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            rgba = rgb + (alpha,)
            
            # Draw activity circle
            bbox = [
                center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius
            ]
            draw.ellipse(bbox, fill=rgba, outline=rgb, width=2)
            
            # Add activity percentage text
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
            except:
                font = ImageFont.load_default()
                
            percentage = f"{activity*100:.0f}%"
            draw.text(center, percentage, fill='#2C3E50', font=font, anchor="mm")
        
        return img
    
    def image_to_base64(self, img):
        """Convert PIL image to base64 string for display in Streamlit"""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

def main():
    st.set_page_config(
        page_title="🧠 Brain Wave Visualizer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .brain-container {
        text-align: center;
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Brain Wave Visualizer</h1>
        <p>Real-time EEG brainwave analysis with brain anatomy visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize visualizer
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = BrainWaveVisualizer()
        st.session_state.collecting = False
        st.session_state.latest_data = None
    
    visualizer = st.session_state.visualizer
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Collection controls
        if st.button("🔴 Start Collection" if not st.session_state.collecting else "⏹️ Stop Collection"):
            if not st.session_state.collecting:
                # Start collection
                visualizer.collector = EEGDataCollector(headset_type="simulation", sample_rate=256)
                visualizer.collector.start_collection()
                st.session_state.collecting = True
                st.success("✅ Data collection started!")
            else:
                # Stop collection
                if visualizer.collector:
                    visualizer.collector.cleanup()
                st.session_state.collecting = False
                st.success("⏹️ Data collection stopped!")
        
        st.markdown("---")
        
        # Display settings
        st.header("📊 Display Settings")
        update_interval = st.slider("Update Interval (seconds)", 0.5, 5.0, 1.0, 0.5)
        show_raw_data = st.checkbox("Show Raw EEG Data", value=True)
        show_frequency_bands = st.checkbox("Show Frequency Bands", value=True)
        show_mental_states = st.checkbox("Show Mental States", value=True)
        
        st.markdown("---")
        
        # Brain wave info
        st.header("🌊 Brainwave Guide")
        st.markdown("""
        **Delta (0.5-4 Hz)**: Deep sleep
        **Theta (4-8 Hz)**: Creativity, meditation
        **Alpha (8-13 Hz)**: Relaxation, focus
        **Beta (13-30 Hz)**: Active thinking, concentration
        **Gamma (30-100 Hz)**: High-level cognitive processing
        """)
    
    # Main content area
    if st.session_state.collecting and visualizer.collector:
        # Auto-refresh
        placeholder = st.empty()
        
        with placeholder.container():
            # Get latest data
            data, timestamps = visualizer.collector.get_latest_data(512)  # 2 seconds of data
            
            if len(data) > 256:  # Need sufficient data for analysis
                eeg_data = np.array(data)
                
                # Perform analysis
                band_powers = visualizer.analyzer.extract_band_power(eeg_data)
                mental_states = visualizer.analyzer.detect_mental_state(eeg_data)
                
                # Create brain visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="brain-container">', unsafe_allow_html=True)
                    st.subheader("🧠 Brain Activity Map")
                    
                    # Create brain image with overlay
                    brain_img = visualizer.create_brain_image()
                    brain_with_activity = visualizer.add_brainwave_overlay(brain_img, band_powers, mental_states)
                    
                    # Display brain image
                    img_b64 = visualizer.image_to_base64(brain_with_activity)
                    st.markdown(
                        f'<img src="data:image/png;base64,{img_b64}" style="width: 100%; max-width: 500px;">',
                        unsafe_allow_html=True
                    )
                    
                    # Legend
                    st.markdown("""
                    **Activity Legend:**
                    - 🔴 **Frontal**: Beta waves (Concentration)
                    - 🟡 **Parietal**: Alpha waves (Relaxation) 
                    - 🔵 **Temporal**: Theta waves (Creativity)
                    - 🟢 **Occipital**: Alpha waves (Visual processing)
                    - 🟣 **Central**: Gamma waves (Motor activity)
                    
                    *Circle size and intensity represent activity level*
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Real-time metrics
                    st.subheader("📊 Live Metrics")
                    
                    # Band powers
                    for band, power in band_powers.items():
                        avg_power = np.mean(power)
                        st.metric(
                            label=f"{band.capitalize()} Power",
                            value=f"{avg_power:.2f} µV²",
                            delta=f"{avg_power/10:.2f}"
                        )
                    
                    st.markdown("---")
                    
                    # Mental states
                    st.subheader("🧠 Mental States")
                    for state, value in mental_states.items():
                        progress = min(value, 3.0) / 3.0  # Normalize to 0-1
                        st.progress(progress)
                        st.caption(f"{state.replace('_', ' ').title()}: {value:.2f}")
                
                # Additional visualizations
                if show_frequency_bands:
                    st.subheader("🌊 Frequency Band Analysis")
                    
                    # Create band power chart
                    bands = list(band_powers.keys())
                    powers = [np.mean(band_powers[band]) for band in bands]
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                    
                    fig = go.Figure(data=[
                        go.Bar(x=bands, y=powers, marker_color=colors)
                    ])
                    fig.update_layout(
                        title="Current Brainwave Power Distribution",
                        xaxis_title="Frequency Band",
                        yaxis_title="Power (µV²)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_raw_data:
                    st.subheader("📈 Raw EEG Signal")
                    
                    # Show last 2 seconds of data
                    recent_data = eeg_data[-512:] if len(eeg_data) > 512 else eeg_data
                    time_axis = np.arange(len(recent_data)) / 256
                    
                    fig = go.Figure()
                    for ch in range(min(4, recent_data.shape[1])):
                        fig.add_trace(go.Scatter(
                            x=time_axis,
                            y=recent_data[:, ch],
                            mode='lines',
                            name=f'Channel {ch+1}',
                            line=dict(width=1)
                        ))
                    
                    fig.update_layout(
                        title="Real-time EEG Signal (Last 2 seconds)",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude (µV)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_mental_states:
                    st.subheader("🎯 Mental State Radar")
                    
                    # Create radar chart for mental states
                    states = list(mental_states.keys())
                    values = list(mental_states.values())
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=states,
                        fill='toself',
                        name='Current State'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(3, max(values))]
                            )),
                        showlegend=True,
                        title="Mental State Profile",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("📡 Collecting data... Please wait for sufficient samples.")
        
        # Auto-refresh
        time.sleep(update_interval)
        st.rerun()
    
    else:
        # Not collecting - show demo
        st.info("🔴 Click 'Start Collection' in the sidebar to begin real-time brainwave visualization!")
        
        # Show static demo brain
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="brain-container">', unsafe_allow_html=True)
            st.subheader("🧠 Brain Activity Map (Demo)")
            
            # Create demo brain image
            brain_img = visualizer.create_brain_image()
            demo_band_powers = {
                'alpha': [25.0], 'beta': [15.0], 'theta': [10.0], 
                'gamma': [5.0], 'delta': [8.0]
            }
            demo_mental_states = {
                'relaxation': 1.5, 'concentration': 1.2, 'meditation': 0.8,
                'alertness': 1.0, 'drowsiness': 0.3
            }
            
            brain_with_demo = visualizer.add_brainwave_overlay(brain_img, demo_band_powers, demo_mental_states)
            img_b64 = visualizer.image_to_base64(brain_with_demo)
            st.markdown(
                f'<img src="data:image/png;base64,{img_b64}" style="width: 100%; max-width: 500px;">',
                unsafe_allow_html=True
            )
            st.caption("*Demo visualization - Start collection for real-time data*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Features")
            st.markdown("""
            ✅ **Real-time EEG monitoring**
            ✅ **Brain anatomy visualization**  
            ✅ **5 frequency band analysis**
            ✅ **Mental state detection**
            ✅ **Interactive controls**
            ✅ **Customizable display**
            """)

if __name__ == "__main__":
    main()
