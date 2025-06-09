#!/bin/bash

# EEG Brainwave System Setup Script
# Installs all required dependencies and sets up the development environment

echo "🧠 EEG Brainwave Analysis System Setup"
echo "====================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    echo "Please install pip3"
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment (optional but recommended)
read -p "🤔 Do you want to create a virtual environment? (y/n): " create_venv

if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    echo "✅ Virtual environment created and activated"
fi

# Upgrade pip
echo "📦 Upgrading pip..."
pip3 install --upgrade pip

# Install core scientific computing packages first
echo "📦 Installing core scientific packages..."
pip3 install numpy scipy matplotlib pandas

# Install EEG-specific packages
echo "📦 Installing EEG analysis packages..."
pip3 install mne scikit-learn

# Install streaming and interface packages
echo "📦 Installing streaming packages..."
pip3 install pylsl || echo "⚠️  pylsl installation failed - some streaming features may not work"
pip3 install muselsl || echo "⚠️  muselsl installation failed - Muse support may not work"
pip3 install pyserial || echo "⚠️  pyserial installation failed - serial headset support may not work"

# Install web interface packages
echo "📦 Installing web interface packages..."
pip3 install streamlit plotly dash

# Install machine learning packages
echo "📦 Installing machine learning packages..."
pip3 install tensorflow torch || echo "⚠️  Deep learning packages installation failed - using scikit-learn only"

# Install utility packages
echo "📦 Installing utility packages..."
pip3 install psutil h5py pywavelets antropy

# Install all remaining requirements
echo "📦 Installing remaining requirements from requirements.txt..."
pip3 install -r requirements.txt || echo "⚠️  Some packages from requirements.txt failed to install"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "🚀 Quick start options:"
echo "1. Test the system:     python3 test_eeg_system.py"
echo "2. Setup headset:       python3 headset_setup.py"
echo "3. Launch dashboard:    streamlit run dashboard.py"
echo "4. Try BCI apps:        python3 bci_applications.py"
echo ""
echo "📚 Read README.md for detailed documentation"

# Check if we can import key modules
echo "🔍 Testing imports..."
python3 -c "
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    print('✅ Core packages working')
except Exception as e:
    print(f'❌ Core package error: {e}')

try:
    from eeg_collector import EEGDataCollector
    from brainwave_analyzer import BrainwaveAnalyzer
    print('✅ EEG modules working')
except Exception as e:
    print(f'❌ EEG module error: {e}')

try:
    import streamlit
    import plotly
    print('✅ Dashboard packages working')
except Exception as e:
    print(f'⚠️  Dashboard package error: {e}')
"

echo ""
echo "✨ You're ready to start analyzing brainwaves!"
