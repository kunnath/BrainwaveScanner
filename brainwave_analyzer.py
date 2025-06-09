"""
EEG Signal Analyzer
Advanced brainwave analysis with frequency domain analysis, feature extraction, and ML classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class BrainwaveAnalyzer:
    """
    Comprehensive brainwave analysis toolkit
    """
    
    def __init__(self, sample_rate: int = 256):
        self.sample_rate = sample_rate
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Initialize ML model
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def filter_signal(self, data: np.ndarray, low_freq: float, high_freq: float, 
                     filter_type: str = 'bandpass') -> np.ndarray:
        """Apply frequency filtering to EEG signal"""
        nyquist = self.sample_rate / 2
        
        if filter_type == 'bandpass':
            low = low_freq / nyquist
            high = high_freq / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
        elif filter_type == 'lowpass':
            high = high_freq / nyquist
            b, a = signal.butter(4, high, btype='low')
        elif filter_type == 'highpass':
            low = low_freq / nyquist
            b, a = signal.butter(4, low, btype='high')
        else:
            raise ValueError("Invalid filter type")
            
        # Apply zero-phase filtering
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        return filtered_data
    
    def remove_artifacts(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove artifacts using statistical thresholding"""
        cleaned_data = data.copy()
        
        # Remove samples that exceed threshold standard deviations
        for ch in range(data.shape[1] if len(data.shape) > 1 else 1):
            if len(data.shape) > 1:
                channel_data = data[:, ch]
            else:
                channel_data = data
                
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            # Find outliers
            outliers = np.abs(channel_data - mean_val) > threshold * std_val
            
            # Replace outliers with interpolated values
            if len(data.shape) > 1:
                cleaned_data[outliers, ch] = np.interp(
                    np.where(outliers)[0],
                    np.where(~outliers)[0],
                    channel_data[~outliers]
                )
            else:
                cleaned_data[outliers] = np.interp(
                    np.where(outliers)[0],
                    np.where(~outliers)[0],
                    channel_data[~outliers]
                )
                
        return cleaned_data
    
    def compute_power_spectrum(self, data: np.ndarray, window_size: int = 256) -> tuple:
        """Compute power spectral density"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        power_spectra = []
        freqs = None
        
        for ch in range(data.shape[1]):
            # Use Welch's method for better frequency resolution
            f, psd = signal.welch(data[:, ch], self.sample_rate, 
                                nperseg=window_size, noverlap=window_size//2)
            power_spectra.append(psd)
            if freqs is None:
                freqs = f  # Use frequencies from Welch's method
            
        return freqs, np.array(power_spectra).T
    
    def extract_band_power(self, data: np.ndarray) -> dict:
        """Extract power in different frequency bands"""
        freqs, power_spectrum = self.compute_power_spectrum(data)
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            if np.any(freq_mask):
                # Calculate average power in this band
                band_power = np.mean(power_spectrum[freq_mask], axis=0)
                band_powers[band_name] = band_power
            else:
                band_powers[band_name] = np.zeros(power_spectrum.shape[1])
                
        return band_powers
    
    def calculate_asymmetry(self, left_channel: np.ndarray, right_channel: np.ndarray) -> dict:
        """Calculate hemispheric asymmetry indices"""
        asymmetry = {}
        
        # Extract band powers for each hemisphere
        left_bands = self.extract_band_power(left_channel.reshape(-1, 1))
        right_bands = self.extract_band_power(right_channel.reshape(-1, 1))
        
        for band_name in self.frequency_bands.keys():
            left_power = left_bands[band_name][0]
            right_power = right_bands[band_name][0]
            
            # Calculate asymmetry index: (right - left) / (right + left)
            if left_power + right_power > 0:
                asymmetry[f'{band_name}_asymmetry'] = (right_power - left_power) / (right_power + left_power)
            else:
                asymmetry[f'{band_name}_asymmetry'] = 0.0
                
        return asymmetry
    
    def extract_features(self, data: np.ndarray) -> dict:
        """Extract comprehensive feature set from EEG data"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(data, axis=0)
        features['std'] = np.std(data, axis=0)
        features['skewness'] = self._skewness(data)
        features['kurtosis'] = self._kurtosis(data)
        
        # Band power features
        band_powers = self.extract_band_power(data)
        for band_name, powers in band_powers.items():
            features[f'{band_name}_power'] = powers
            
        # Relative band powers
        total_power = sum(band_powers.values())
        for band_name, powers in band_powers.items():
            rel_power = powers / (total_power + 1e-10)  # Avoid division by zero
            features[f'{band_name}_relative'] = rel_power
            
        # Band power ratios
        if 'alpha' in band_powers and 'beta' in band_powers:
            features['alpha_beta_ratio'] = band_powers['alpha'] / (band_powers['beta'] + 1e-10)
        if 'theta' in band_powers and 'alpha' in band_powers:
            features['theta_alpha_ratio'] = band_powers['theta'] / (band_powers['alpha'] + 1e-10)
            
        # Spectral edge frequency (frequency below which 90% of power is contained)
        freqs, power_spectrum = self.compute_power_spectrum(data)
        cumulative_power = np.cumsum(power_spectrum, axis=0)
        total_power_spectrum = cumulative_power[-1]
        
        features['spectral_edge_90'] = []
        for ch in range(power_spectrum.shape[1]):
            edge_idx = np.where(cumulative_power[:, ch] >= 0.9 * total_power_spectrum[ch])[0]
            if len(edge_idx) > 0:
                features['spectral_edge_90'].append(freqs[edge_idx[0]])
            else:
                features['spectral_edge_90'].append(freqs[-1])
                
        features['spectral_edge_90'] = np.array(features['spectral_edge_90'])
        
        return features
    
    def _skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness"""
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0)
        return np.mean(((data - mean_val) / (std_val + 1e-10))**3, axis=0)
    
    def _kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis"""
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0)
        return np.mean(((data - mean_val) / (std_val + 1e-10))**4, axis=0) - 3
    
    def detect_mental_state(self, data: np.ndarray) -> dict:
        """Detect mental states based on brainwave patterns"""
        band_powers = self.extract_band_power(data)
        
        # Calculate relative powers
        total_power = sum([np.sum(power) for power in band_powers.values()])
        
        states = {}
        
        if total_power > 0:
            # Relaxation (high alpha, low beta)
            alpha_ratio = np.sum(band_powers['alpha']) / total_power
            beta_ratio = np.sum(band_powers['beta']) / total_power
            states['relaxation'] = alpha_ratio / (beta_ratio + 0.1)
            
            # Concentration (high beta, low theta)
            theta_ratio = np.sum(band_powers['theta']) / total_power
            states['concentration'] = beta_ratio / (theta_ratio + 0.1)
            
            # Meditation (high theta, low beta)
            states['meditation'] = theta_ratio / (beta_ratio + 0.1)
            
            # Alertness (high beta and gamma)
            gamma_ratio = np.sum(band_powers['gamma']) / total_power if 'gamma' in band_powers else 0
            states['alertness'] = (beta_ratio + gamma_ratio) / 2
            
            # Drowsiness (high delta and theta)
            delta_ratio = np.sum(band_powers['delta']) / total_power
            states['drowsiness'] = (delta_ratio + theta_ratio) / 2
        else:
            # Default values if no signal
            for state in ['relaxation', 'concentration', 'meditation', 'alertness', 'drowsiness']:
                states[state] = 0.0
        
        return states
    
    def train_classifier(self, training_data: list, labels: list):
        """Train ML classifier for EEG pattern recognition"""
        features_list = []
        
        for data_segment in training_data:
            features = self.extract_features(data_segment)
            # Flatten all features into a single vector
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value.flatten())
                else:
                    feature_vector.append(value)
            features_list.append(feature_vector)
        
        # Prepare training data
        X = np.array(features_list)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_scaled)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
    def predict_state(self, data: np.ndarray) -> str:
        """Predict mental state using trained classifier"""
        if not self.is_trained:
            return "classifier_not_trained"
            
        features = self.extract_features(data)
        feature_vector = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                feature_vector.extend(value.flatten())
            else:
                feature_vector.append(value)
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.classifier.predict(X_scaled)[0]
        confidence = np.max(self.classifier.predict_proba(X_scaled))
        
        return f"{prediction} (confidence: {confidence:.2f})"
    
    def plot_analysis(self, data: np.ndarray, title: str = "EEG Analysis"):
        """Create comprehensive analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Raw signal
        axes[0, 0].plot(data[:1000])  # Plot first 1000 samples
        axes[0, 0].set_title('Raw EEG Signal')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude (µV)')
        
        # Power spectrum
        freqs, power_spectrum = self.compute_power_spectrum(data)
        for ch in range(min(4, power_spectrum.shape[1])):  # Plot up to 4 channels
            axes[0, 1].semilogy(freqs, power_spectrum[:, ch], label=f'Channel {ch+1}')
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power (µV²/Hz)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Band powers
        band_powers = self.extract_band_power(data)
        bands = list(band_powers.keys())
        powers = [np.mean(band_powers[band]) for band in bands]
        
        axes[0, 2].bar(bands, powers)
        axes[0, 2].set_title('Average Band Powers')
        axes[0, 2].set_ylabel('Power (µV²)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Mental states
        states = self.detect_mental_state(data)
        state_names = list(states.keys())
        state_values = list(states.values())
        
        axes[1, 0].bar(state_names, state_values)
        axes[1, 0].set_title('Mental State Indicators')
        axes[1, 0].set_ylabel('Index Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Spectrogram
        if len(data.shape) > 1 and data.shape[1] > 0:
            f, t, Sxx = signal.spectrogram(data[:, 0], self.sample_rate, nperseg=256)
            im = axes[1, 1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            axes[1, 1].set_title('Spectrogram (Channel 1)')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Frequency (Hz)')
            axes[1, 1].set_ylim(0, 50)  # Focus on relevant frequencies
            plt.colorbar(im, ax=axes[1, 1], label='Power (dB)')
        
        # Band power over time
        window_size = 256
        hop_size = 128
        time_points = []
        band_powers_time = {band: [] for band in self.frequency_bands.keys()}
        
        for i in range(0, len(data) - window_size, hop_size):
            window_data = data[i:i+window_size]
            bp = self.extract_band_power(window_data)
            time_points.append(i / self.sample_rate)
            
            for band in self.frequency_bands.keys():
                band_powers_time[band].append(np.mean(bp[band]))
        
        for band in self.frequency_bands.keys():
            axes[1, 2].plot(time_points, band_powers_time[band], label=band)
        
        axes[1, 2].set_title('Band Powers Over Time')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Power (µV²)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Test the analyzer with simulated data
    print("🧠 Brainwave Analyzer Test")
    print("=" * 40)
    
    analyzer = BrainwaveAnalyzer(sample_rate=256)
    
    # Generate test data
    duration = 10  # seconds
    samples = duration * 256
    t = np.linspace(0, duration, samples)
    
    # Simulate multi-channel EEG with different brainwave activities
    channels = 4
    test_data = np.zeros((samples, channels))
    
    for ch in range(channels):
        # Mix of different frequencies
        alpha_wave = np.sin(2 * np.pi * 10 * t) * 0.5
        beta_wave = np.sin(2 * np.pi * 20 * t) * 0.3
        theta_wave = np.sin(2 * np.pi * 6 * t) * 0.2
        noise = np.random.normal(0, 0.1, samples)
        
        test_data[:, ch] = alpha_wave + beta_wave + theta_wave + noise
    
    # Test analysis functions
    print("\n1. Extracting features...")
    features = analyzer.extract_features(test_data)
    print(f"Extracted {len(features)} feature types")
    
    print("\n2. Detecting mental states...")
    states = analyzer.detect_mental_state(test_data)
    for state, value in states.items():
        print(f"{state}: {value:.3f}")
    
    print("\n3. Band power analysis...")
    band_powers = analyzer.extract_band_power(test_data)
    for band, powers in band_powers.items():
        print(f"{band}: {np.mean(powers):.3f} µV²")
    
    print("\n4. Creating analysis plots...")
    fig = analyzer.plot_analysis(test_data, "Test EEG Analysis")
    plt.savefig('/Users/kunnath/Projects/brainwave/test_analysis.png', dpi=300, bbox_inches='tight')
    print("Analysis plot saved as test_analysis.png")
    
    plt.show()
    print("\nAnalyzer test completed!")
