# Regular Headset to Neurable-Style EEG Guide

This guide explains how to use regular EEG headsets (Muse, Emotiv, OpenBCI, etc.) to implement advanced brain-computer interface techniques similar to commercial systems like Neurable Enten headphones.

## 🎯 What We Can Achieve

### 1. **Enhanced Signal Processing**
- **Commercial-grade filtering**: Multi-stage noise reduction, notch filters for power line interference
- **Advanced artifact removal**: ICA-based eye movement and muscle artifact removal
- **Adaptive filtering**: Wiener filters for dynamic noise reduction

### 2. **Intention Detection**
- **Focus vs. Relaxation**: Detect concentration states vs. relaxed states
- **Selection Intent**: Recognize when user intends to "click" or select something
- **Navigation Intent**: Basic directional intent detection
- **Confirmation Intent**: Detect yes/no or confirm/cancel intentions

### 3. **Advanced Feature Extraction**
- **Spectral Features**: Detailed frequency band analysis (sub-bands of alpha, beta, etc.)
- **Temporal Features**: Statistical measures, Hjorth parameters
- **Connectivity Features**: Inter-channel correlation and phase relationships
- **Nonlinear Features**: Fractal dimension, entropy measures

## 🎧 Supported Headset Types

### 1. **Muse Headband**
```python
# Configuration for Muse
electrode_positions = ['TP9', 'AF7', 'AF8', 'TP10']
focus_channels = [1, 2]      # AF7, AF8 (frontal)
relaxation_channels = [0, 3]  # TP9, TP10 (temporal)
```

### 2. **Emotiv EPOC**
```python
# Configuration for Emotiv
electrode_positions = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
focus_channels = [2, 11]     # F3, F4 (frontal)
relaxation_channels = [6, 7] # O1, O2 (occipital)
```

### 3. **OpenBCI**
```python
# Configuration for OpenBCI
electrode_positions = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
focus_channels = [0, 1]      # Fp1, Fp2 (frontal)
relaxation_channels = [6, 7] # O1, O2 (occipital)
```

## 🚀 Quick Start Guide

### 1. **Installation**
```bash
# Install required packages
pip install numpy scipy scikit-learn matplotlib pandas joblib

# Install EEG-specific packages
pip install mne pylsl pyserial

# For specific headsets
pip install muselsl        # For Muse
pip install pyOpenBCI      # For OpenBCI
```

### 2. **Basic Usage**
```python
from neurable_adapter import NeurableStyleAdapter

# Initialize adapter for your headset
adapter = NeurableStyleAdapter(headset_type="muse", sample_rate=256)

# Perform calibration (required once per user)
adapter.calibrate_for_regular_headset("muse")

# Start real-time intention detection
adapter.start_real_time_processing("muse")

# Get intentions
while True:
    intention = adapter.get_latest_intention()
    if intention:
        print(f"Detected: {intention['prediction']} (confidence: {intention['confidence']:.2f})")
```

### 3. **Advanced Processing Test**
```python
from advanced_eeg_processor import AdvancedEEGProcessor

# Initialize processor
processor = AdvancedEEGProcessor(sample_rate=256, channels=4)

# Test with your data
filtered_data = processor.apply_commercial_grade_filtering(raw_data)
clean_data = processor.remove_artifacts_advanced(filtered_data)
features = processor.extract_advanced_features(clean_data)
```

## 🧠 Key Techniques Implemented

### 1. **Multi-Stage Filtering**
```python
# Power line noise removal (50/60 Hz)
notch_filter(data, frequencies=[50, 60])

# High-pass filter (remove DC drift)
high_pass_filter(data, cutoff=0.1)

# Low-pass filter (remove high-frequency noise)
low_pass_filter(data, cutoff=100)

# Adaptive Wiener filter
wiener_filter(data, noise_estimation="adaptive")
```

### 2. **ICA Artifact Removal**
```python
# Separate signal components
ica_components = ICA.fit_transform(data)

# Identify artifacts (high kurtosis, low frequency)
artifact_components = detect_artifacts(ica_components)

# Remove artifacts and reconstruct
clean_data = ICA.inverse_transform(remove_components(ica_components, artifacts))
```

### 3. **Advanced Feature Engineering**
```python
features = {
    # Spectral features
    'alpha_low_power': extract_band_power(data, 8, 10),
    'alpha_high_power': extract_band_power(data, 10, 13),
    'beta_low_power': extract_band_power(data, 13, 20),
    
    # Temporal features
    'hjorth_activity': calculate_hjorth_activity(data),
    'hjorth_mobility': calculate_hjorth_mobility(data),
    'hjorth_complexity': calculate_hjorth_complexity(data),
    
    # Nonlinear features
    'approximate_entropy': calculate_approximate_entropy(data),
    'fractal_dimension': calculate_fractal_dimension(data),
    
    # Connectivity features
    'channel_correlation': calculate_cross_correlation(data),
    'phase_lag_index': calculate_phase_lag_index(data)
}
```

## 🎮 Practical Applications

### 1. **Focus Training Application**
```python
def focus_training_app():
    adapter = NeurableStyleAdapter(headset_type="muse")
    adapter.calibrate_for_regular_headset("muse")
    
    while True:
        intention = adapter.get_latest_intention()
        if intention and intention['prediction'] == 'focus_increase':
            print("🎯 Great focus! Keep it up!")
            update_focus_score(intention['confidence'])
```

### 2. **Meditation Assistant**
```python
def meditation_assistant():
    adapter = NeurableStyleAdapter(headset_type="muse")
    
    while meditating:
        intention = adapter.get_latest_intention()
        if intention and intention['prediction'] == 'focus_decrease':
            print("😌 Deep relaxation detected")
            play_calming_sound()
```

### 3. **Simple BCI Control**
```python
def simple_bci_control():
    adapter = NeurableStyleAdapter(headset_type="emotiv")
    
    while True:
        intention = adapter.get_latest_intention()
        
        if intention['prediction'] == 'select_object':
            mouse_click()
        elif intention['prediction'] == 'navigate_left':
            move_cursor_left()
        elif intention['prediction'] == 'navigate_right':
            move_cursor_right()
```

## 📊 Calibration Process

### 1. **Baseline Recording**
- **Duration**: 10 seconds
- **Instructions**: "Close your eyes and relax"
- **Purpose**: Establish individual baseline patterns

### 2. **Focus Task**
- **Duration**: 10 seconds
- **Instructions**: "Focus on mental math (e.g., 17 × 23)"
- **Purpose**: Capture focused attention patterns

### 3. **Relaxation Task**
- **Duration**: 10 seconds
- **Instructions**: "Think of a peaceful scene"
- **Purpose**: Capture relaxed state patterns

### 4. **Selection Task**
- **Duration**: 10 seconds
- **Instructions**: "Imagine clicking/selecting repeatedly"
- **Purpose**: Capture action intention patterns

## 🔧 Optimization Tips

### 1. **Improve Signal Quality**
```python
# Ensure good electrode contact
check_impedance(headset)

# Reduce environmental noise
use_shielded_room()
minimize_electrical_interference()

# Optimize electrode placement
follow_10_20_system()
use_conductive_gel()
```

### 2. **Enhance Classification**
```python
# Use ensemble methods
classifiers = {
    'random_forest': RandomForestClassifier(n_estimators=200),
    'gradient_boosting': GradientBoostingClassifier(),
    'svm': SVC(kernel='rbf', probability=True)
}

# Feature selection
selected_features = select_best_features(all_features, target_labels)

# Cross-validation
cv_scores = cross_validate(classifier, features, labels, cv=5)
```

### 3. **Real-time Performance**
```python
# Optimize processing pipeline
use_efficient_algorithms()
implement_sliding_windows()
cache_computed_features()

# Parallel processing
use_multiprocessing()
implement_async_processing()
```

## 📈 Expected Performance

### 1. **Classification Accuracy**
- **Focus vs. Relaxation**: 70-85% accuracy
- **Selection Intent**: 60-75% accuracy
- **Multi-class (4+ states)**: 50-70% accuracy

### 2. **Response Time**
- **Processing Delay**: 500ms - 2 seconds
- **Real-time Processing**: 2-5 Hz update rate
- **Calibration Time**: 5-10 minutes

### 3. **Individual Variability**
- **Some users**: 80%+ accuracy
- **Average users**: 65-75% accuracy
- **Challenging users**: 50-65% accuracy

## ⚠️ Limitations vs. Commercial Systems

### **What Regular Headsets CAN'T Do:**
1. **Precise Spatial Resolution**: Limited electrode coverage
2. **High-Speed BCIs**: Slower than research-grade systems
3. **Complex Commands**: Limited to basic intention categories
4. **Universal Compatibility**: Requires individual calibration

### **What We CAN Achieve:**
1. **Basic Intent Detection**: Focus, relaxation, selection
2. **Meditation/Focus Training**: Real-time feedback
3. **Simple Control**: Basic navigation and selection
4. **Research Applications**: Feature extraction and analysis

## 🎯 Future Enhancements

### 1. **Advanced Algorithms**
- Deep learning models (CNN, LSTM)
- Transfer learning between users
- Adaptive recalibration

### 2. **Better Hardware Integration**
- Multi-headset fusion
- Improved artifact removal
- Real-time impedance monitoring

### 3. **Application Development**
- VR/AR integration
- Gaming applications
- Assistive technology

## 📚 References

1. **Neurable Technology**: Commercial EEG-based BCI
2. **OpenBCI**: Open-source brain-computer interfaces
3. **MNE-Python**: EEG analysis toolkit
4. **scikit-learn**: Machine learning library

## 🚀 Getting Started

Run the demonstration:
```bash
cd /Users/kunnath/Projects/brainwave
python3 neurable_adapter.py
```

This will walk you through the complete process of calibration and real-time intention detection using simulated EEG data that mimics real headset behavior.
