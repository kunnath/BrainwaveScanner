"""
Enhanced EEG System with Neurable-inspired Techniques
Implements advanced signal processing, artifact removal, and intention detection
similar to commercial-grade EEG headsets like Neurable Enten
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedEEGProcessor:
    """
    Advanced EEG processing system with commercial-grade techniques
    Inspired by Neurable Enten and similar high-end EEG devices
    """
    
    def __init__(self, sample_rate=256, channels=4):
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Enhanced frequency bands with sub-bands
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha_low': (8, 10),
            'alpha_high': (10, 13),
            'beta_low': (13, 20),
            'beta_high': (20, 30),
            'gamma_low': (30, 50),
            'gamma_high': (50, 100)
        }
        
        # Initialize advanced models
        self.ica = FastICA(n_components=channels, random_state=42)
        self.pca = PCA(n_components=channels)
        self.scaler = StandardScaler()
        
        # Multiple classifiers for ensemble learning
        self.classifiers = {
            'rf': RandomForestClassifier(n_estimators=200, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # Intent detection categories (Neurable-style)
        self.intent_categories = {
            'focus_increase': 'Intention to increase focus',
            'focus_decrease': 'Intention to relax',
            'select_object': 'Intention to select/click',
            'navigate_left': 'Intention to move left',
            'navigate_right': 'Intention to move right',
            'navigate_up': 'Intention to move up',
            'navigate_down': 'Intention to move down',
            'confirm_action': 'Intention to confirm',
            'cancel_action': 'Intention to cancel'
        }
        
        self.is_trained = False
        self.calibration_data = []
        self.baseline_features = None
        
    def apply_commercial_grade_filtering(self, data):
        """
        Apply multi-stage filtering similar to commercial EEG devices
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        filtered_data = data.copy()
        
        # 1. Notch filter for power line interference (50/60 Hz)
        for freq in [50, 60]:  # Remove both EU and US power line frequencies
            b_notch, a_notch = signal.iirnotch(freq, Q=30, fs=self.sample_rate)
            filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=0)
        
        # 2. High-pass filter to remove DC drift and very low frequencies
        high_cutoff = 0.1
        sos_high = signal.butter(4, high_cutoff, btype='high', fs=self.sample_rate, output='sos')
        filtered_data = signal.sosfiltfilt(sos_high, filtered_data, axis=0)
        
        # 3. Low-pass filter to remove high-frequency noise
        low_cutoff = 100
        sos_low = signal.butter(4, low_cutoff, btype='low', fs=self.sample_rate, output='sos')
        filtered_data = signal.sosfiltfilt(sos_low, filtered_data, axis=0)
        
        # 4. Adaptive noise reduction using Wiener filter approach
        filtered_data = self._apply_wiener_filter(filtered_data)
        
        return filtered_data
    
    def _apply_wiener_filter(self, data):
        """
        Apply Wiener filter for adaptive noise reduction
        """
        filtered_data = data.copy()
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Estimate noise level from high-frequency components
            high_freq_data = signal.filtfilt(*signal.butter(4, 30, btype='high', fs=self.sample_rate), 
                                           channel_data)
            noise_var = np.var(high_freq_data)
            signal_var = np.var(channel_data)
            
            # Wiener filter coefficient
            wiener_coeff = signal_var / (signal_var + noise_var + 1e-10)
            
            # Apply filter
            filtered_data[:, ch] = channel_data * wiener_coeff
            
        return filtered_data
    
    def remove_artifacts_advanced(self, data):
        """
        Advanced artifact removal using ICA and statistical methods
        Similar to commercial EEG systems
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        if data.shape[1] < 2:
            # Simple artifact removal for single channel
            return self._remove_artifacts_single_channel(data)
        
        # 1. Independent Component Analysis for artifact separation
        try:
            # ICA requires sufficient data points
            if data.shape[0] >= data.shape[1] * 20:
                ica_data = self.ica.fit_transform(data.T).T
                
                # Identify artifact components (high kurtosis, low frequency content)
                artifact_components = []
                for i in range(ica_data.shape[1]):
                    component = ica_data[:, i]
                    
                    # Calculate kurtosis (artifacts typically have high kurtosis)
                    kurt = self._kurtosis(component)
                    
                    # Calculate frequency content
                    freqs, psd = signal.welch(component, self.sample_rate)
                    low_freq_power = np.sum(psd[freqs < 4])  # Delta range
                    total_power = np.sum(psd)
                    low_freq_ratio = low_freq_power / (total_power + 1e-10)
                    
                    # Mark as artifact if high kurtosis and high low-frequency content
                    if abs(kurt) > 5 or low_freq_ratio > 0.6:
                        artifact_components.append(i)
                
                # Remove artifact components
                if artifact_components:
                    ica_data[:, artifact_components] = 0
                    
                # Transform back to original space
                cleaned_data = self.ica.inverse_transform(ica_data.T).T
            else:
                cleaned_data = data
                
        except Exception as e:
            print(f"ICA artifact removal failed: {e}")
            cleaned_data = data
        
        # 2. Statistical outlier removal
        cleaned_data = self._remove_statistical_outliers(cleaned_data)
        
        # 3. Gradient artifact removal (for EEG during movement)
        cleaned_data = self._remove_gradient_artifacts(cleaned_data)
        
        return cleaned_data
    
    def _remove_artifacts_single_channel(self, data):
        """Artifact removal for single channel data"""
        cleaned_data = data.copy()
        
        # Z-score based outlier removal
        z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-10))
        outliers = z_scores > 3
        
        # Interpolate outliers
        if np.any(outliers):
            cleaned_data[outliers] = np.interp(
                np.where(outliers)[0],
                np.where(~outliers)[0],
                data[~outliers]
            )
        
        return cleaned_data
    
    def _remove_statistical_outliers(self, data):
        """Remove statistical outliers using robust methods"""
        cleaned_data = data.copy()
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Use median absolute deviation for robust outlier detection
            median = np.median(channel_data)
            mad = np.median(np.abs(channel_data - median))
            
            # Modified z-score
            modified_z_scores = 0.6745 * (channel_data - median) / (mad + 1e-10)
            outliers = np.abs(modified_z_scores) > 3.5
            
            # Interpolate outliers
            if np.any(outliers):
                cleaned_data[outliers, ch] = np.interp(
                    np.where(outliers)[0],
                    np.where(~outliers)[0],
                    channel_data[~outliers]
                )
        
        return cleaned_data
    
    def _remove_gradient_artifacts(self, data):
        """Remove gradient artifacts caused by movement"""
        cleaned_data = data.copy()
        
        # Detect rapid changes (gradients)
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Calculate gradient
            gradient = np.gradient(channel_data)
            
            # Detect sudden changes
            gradient_threshold = 3 * np.std(gradient)
            sudden_changes = np.abs(gradient) > gradient_threshold
            
            # Apply smoothing to sudden changes
            if np.any(sudden_changes):
                # Use a moving average to smooth sudden changes
                window_size = 5
                for i in np.where(sudden_changes)[0]:
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(channel_data), i + window_size // 2 + 1)
                    
                    # Replace with local average (excluding the artifact point)
                    local_data = np.concatenate([
                        channel_data[start_idx:i],
                        channel_data[i+1:end_idx]
                    ])
                    
                    if len(local_data) > 0:
                        cleaned_data[i, ch] = np.mean(local_data)
        
        return cleaned_data
    
    def extract_advanced_features(self, data):
        """
        Extract comprehensive feature set similar to commercial EEG systems
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        features = {}
        
        # 1. Enhanced spectral features
        spectral_features = self._extract_spectral_features(data)
        features.update(spectral_features)
        
        # 2. Temporal features
        temporal_features = self._extract_temporal_features(data)
        features.update(temporal_features)
        
        # 3. Connectivity features (if multi-channel)
        if data.shape[1] > 1:
            connectivity_features = self._extract_connectivity_features(data)
            features.update(connectivity_features)
        
        # 4. Nonlinear features
        nonlinear_features = self._extract_nonlinear_features(data)
        features.update(nonlinear_features)
        
        # 5. Event-related features
        event_features = self._extract_event_related_features(data)
        features.update(event_features)
        
        return features
    
    def _extract_spectral_features(self, data):
        """Extract advanced spectral features"""
        features = {}
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Power spectral density
            freqs, psd = signal.welch(channel_data, self.sample_rate, nperseg=256)
            
            # Band powers for enhanced bands
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(freq_mask):
                    band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                    features[f'{band_name}_power_ch{ch}'] = band_power
            
            # Spectral edge frequency (95%)
            cumsum_psd = np.cumsum(psd)
            total_power = cumsum_psd[-1]
            edge_95_idx = np.where(cumsum_psd >= 0.95 * total_power)[0]
            if len(edge_95_idx) > 0:
                features[f'spectral_edge_95_ch{ch}'] = freqs[edge_95_idx[0]]
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
            features[f'spectral_centroid_ch{ch}'] = spectral_centroid
            
            # Spectral rolloff
            rolloff_threshold = 0.85
            rolloff_idx = np.where(cumsum_psd >= rolloff_threshold * total_power)[0]
            if len(rolloff_idx) > 0:
                features[f'spectral_rolloff_ch{ch}'] = freqs[rolloff_idx[0]]
        
        return features
    
    def _extract_temporal_features(self, data):
        """Extract temporal domain features"""
        features = {}
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Basic statistics
            features[f'mean_ch{ch}'] = np.mean(channel_data)
            features[f'std_ch{ch}'] = np.std(channel_data)
            features[f'var_ch{ch}'] = np.var(channel_data)
            features[f'skewness_ch{ch}'] = self._skewness(channel_data)
            features[f'kurtosis_ch{ch}'] = self._kurtosis(channel_data)
            
            # Peak-to-peak amplitude
            features[f'peak_to_peak_ch{ch}'] = np.max(channel_data) - np.min(channel_data)
            
            # Root mean square
            features[f'rms_ch{ch}'] = np.sqrt(np.mean(channel_data**2))
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(channel_data)))[0]
            features[f'zero_crossing_rate_ch{ch}'] = len(zero_crossings) / len(channel_data)
            
            # Hjorth parameters
            hjorth_activity = np.var(channel_data)
            features[f'hjorth_activity_ch{ch}'] = hjorth_activity
            
            first_derivative = np.diff(channel_data)
            hjorth_mobility = np.sqrt(np.var(first_derivative) / (hjorth_activity + 1e-10))
            features[f'hjorth_mobility_ch{ch}'] = hjorth_mobility
            
            second_derivative = np.diff(first_derivative)
            hjorth_complexity = np.sqrt(np.var(second_derivative) / (np.var(first_derivative) + 1e-10)) / hjorth_mobility
            features[f'hjorth_complexity_ch{ch}'] = hjorth_complexity
        
        return features
    
    def _extract_connectivity_features(self, data):
        """Extract connectivity features between channels"""
        features = {}
        
        # Cross-correlation between channels
        for i in range(data.shape[1]):
            for j in range(i+1, data.shape[1]):
                correlation = np.corrcoef(data[:, i], data[:, j])[0, 1]
                features[f'correlation_ch{i}_ch{j}'] = correlation
                
                # Phase lag index (simplified version)
                analytic_i = signal.hilbert(data[:, i])
                analytic_j = signal.hilbert(data[:, j])
                phase_diff = np.angle(analytic_i) - np.angle(analytic_j)
                pli = np.abs(np.mean(np.sign(np.imag(np.exp(1j * phase_diff)))))
                features[f'pli_ch{i}_ch{j}'] = pli
        
        return features
    
    def _extract_nonlinear_features(self, data):
        """Extract nonlinear features"""
        features = {}
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Approximate entropy
            features[f'approximate_entropy_ch{ch}'] = self._approximate_entropy(channel_data)
            
            # Fractal dimension (Higuchi method)
            features[f'fractal_dimension_ch{ch}'] = self._higuchi_fractal_dimension(channel_data)
            
            # Sample entropy
            features[f'sample_entropy_ch{ch}'] = self._sample_entropy(channel_data)
        
        return features
    
    def _extract_event_related_features(self, data):
        """Extract event-related features for intention detection"""
        features = {}
        
        # Look for specific patterns that might indicate intentions
        window_size = int(0.5 * self.sample_rate)  # 500ms windows
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Event-related potential-like features
            # Look for characteristic patterns in different time windows
            
            # Early response (0-200ms equivalent in terms of samples)
            early_window = channel_data[:int(0.2 * self.sample_rate)]
            if len(early_window) > 0:
                features[f'early_response_mean_ch{ch}'] = np.mean(early_window)
                features[f'early_response_peak_ch{ch}'] = np.max(np.abs(early_window))
            
            # Late response (200-500ms equivalent)
            late_start = int(0.2 * self.sample_rate)
            late_end = int(0.5 * self.sample_rate)
            if late_end <= len(channel_data):
                late_window = channel_data[late_start:late_end]
                features[f'late_response_mean_ch{ch}'] = np.mean(late_window)
                features[f'late_response_peak_ch{ch}'] = np.max(np.abs(late_window))
        
        return features
    
    def detect_intentions(self, data):
        """
        Detect user intentions similar to Neurable's approach
        """
        if not self.is_trained:
            return {'status': 'not_trained', 'intentions': {}}
        
        # Extract features
        features = self.extract_advanced_features(data)
        
        # Convert to feature vector
        feature_vector = []
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, (int, float)) and not np.isnan(value):
                feature_vector.append(value)
            else:
                feature_vector.append(0.0)
        
        if len(feature_vector) == 0:
            return {'status': 'no_features', 'intentions': {}}
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except:
            return {'status': 'scaling_error', 'intentions': {}}
        
        # Ensemble prediction
        predictions = {}
        confidences = {}
        
        for clf_name, clf in self.classifiers.items():
            if hasattr(clf, 'predict_proba'):
                try:
                    pred = clf.predict(X_scaled)[0]
                    proba = clf.predict_proba(X_scaled)[0]
                    confidence = np.max(proba)
                    
                    predictions[clf_name] = pred
                    confidences[clf_name] = confidence
                except:
                    continue
        
        # Combine predictions (majority vote with confidence weighting)
        if predictions:
            # Weight votes by confidence
            weighted_votes = {}
            for clf_name, pred in predictions.items():
                confidence = confidences[clf_name]
                if pred not in weighted_votes:
                    weighted_votes[pred] = 0
                weighted_votes[pred] += confidence
            
            # Get the prediction with highest weighted vote
            final_prediction = max(weighted_votes.items(), key=lambda x: x[1])
            prediction_label = final_prediction[0]
            prediction_confidence = final_prediction[1] / len(predictions)
            
            return {
                'status': 'success',
                'prediction': prediction_label,
                'confidence': prediction_confidence,
                'individual_predictions': predictions,
                'individual_confidences': confidences
            }
        else:
            return {'status': 'prediction_error', 'intentions': {}}
    
    def calibrate_system(self, calibration_sessions):
        """
        Calibrate the system with user-specific data
        Similar to Neurable's calibration process
        """
        print("Starting advanced EEG system calibration...")
        
        all_features = []
        all_labels = []
        
        for session in calibration_sessions:
            data = session['data']
            label = session['label']
            
            # Preprocess data
            filtered_data = self.apply_commercial_grade_filtering(data)
            clean_data = self.remove_artifacts_advanced(filtered_data)
            
            # Extract features
            features = self.extract_advanced_features(clean_data)
            
            # Convert to feature vector
            feature_vector = []
            for key in sorted(features.keys()):
                value = features[key]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(value)
                else:
                    feature_vector.append(0.0)
            
            if len(feature_vector) > 0:
                all_features.append(feature_vector)
                all_labels.append(label)
        
        if len(all_features) == 0:
            print("❌ No valid features extracted for calibration")
            return False
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"Calibration data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        if len(np.unique(y)) > 1 and len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
        
        # Train all classifiers
        calibration_scores = {}
        
        for clf_name, clf in self.classifiers.items():
            try:
                clf.fit(X_train, y_train)
                
                if len(X_test) > 0:
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    calibration_scores[clf_name] = accuracy
                    print(f"  {clf_name.upper()} accuracy: {accuracy:.3f}")
                
                # Cross-validation score
                if len(X_scaled) > 5:
                    cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(X_scaled)))
                    print(f"  {clf_name.upper()} CV score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed to train {clf_name}: {e}")
                continue
        
        self.is_trained = True
        print("✅ Advanced EEG system calibration completed!")
        
        return True
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'classifiers': self.classifiers,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            
            self.classifiers = model_data['classifiers']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.sample_rate = model_data['sample_rate']
            self.channels = model_data['channels']
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    # Helper methods
    def _skewness(self, data):
        """Calculate skewness"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        return np.mean(((data - mean_val) / (std_val + 1e-10))**3)
    
    def _kurtosis(self, data):
        """Calculate kurtosis"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        return np.mean(((data - mean_val) / (std_val + 1e-10))**4) - 3
    
    def _approximate_entropy(self, data, m=2, r=None):
        """Calculate approximate entropy"""
        if r is None:
            r = 0.2 * np.std(data)
        
        N = len(data)
        
        def _maxdist(xi, xj, N, m):
            return max([abs(ua - va) for ua, va in zip(xi[0:m], xj[0:m])])
        
        def _phi(m):
            patterns = np.array([[data[j] for j in range(i, i + m)] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j], N, m) <= r:
                        C[i] += 1.0
                        
            phi = (N - m + 1.0)**(-1) * sum([np.log(C[i] / (N - m + 1.0)) for i in range(N - m + 1)])
            return phi
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _sample_entropy(self, data, m=2, r=None):
        """Calculate sample entropy"""
        if r is None:
            r = 0.2 * np.std(data)
        
        N = len(data)
        
        def _get_matches(m):
            matches = 0
            for i in range(N - m):
                for j in range(i + 1, N - m):
                    if np.max(np.abs(data[i:i+m] - data[j:j+m])) <= r:
                        matches += 1
            return matches
        
        try:
            A = _get_matches(m)
            B = _get_matches(m + 1)
            
            if A == 0:
                return float('inf')
            else:
                return -np.log(B / A)
        except:
            return 0.0
    
    def _higuchi_fractal_dimension(self, data, k_max=10):
        """Calculate Higuchi fractal dimension"""
        N = len(data)
        L = np.zeros(k_max)
        
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(k):
                Lmk = 0
                for i in range(1, int((N - m) / k)):
                    Lmk += abs(data[m + i * k] - data[m + (i - 1) * k])
                
                if int((N - m) / k) > 0:
                    Lmk = Lmk * (N - 1) / (int((N - m) / k) * k)
                    Lk.append(Lmk)
            
            if Lk:
                L[k - 1] = np.mean(Lk)
        
        # Linear regression in log-log space
        x = np.log(range(1, k_max + 1))
        y = np.log(L)
        
        # Remove infinite or NaN values
        valid_indices = np.isfinite(x) & np.isfinite(y) & (y != 0)
        
        if np.sum(valid_indices) < 2:
            return 1.0
        
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        
        try:
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            return -slope
        except:
            return 1.0


if __name__ == "__main__":
    # Test the advanced EEG processor
    print("🧠 Advanced EEG Processor Test (Neurable-inspired)")
    print("=" * 60)
    
    processor = AdvancedEEGProcessor(sample_rate=256, channels=4)
    
    # Generate test data with different "intentions"
    duration = 2  # seconds
    samples = duration * 256
    t = np.linspace(0, duration, samples)
    
    # Simulate different intention patterns
    test_sessions = []
    
    # Focus intention (high beta activity)
    focus_data = np.zeros((samples, 4))
    for ch in range(4):
        alpha_wave = np.sin(2 * np.pi * 10 * t) * 0.3
        beta_wave = np.sin(2 * np.pi * 20 * t) * 0.8  # High beta for focus
        noise = np.random.normal(0, 0.1, samples)
        focus_data[:, ch] = alpha_wave + beta_wave + noise
    
    test_sessions.append({'data': focus_data, 'label': 'focus_increase'})
    
    # Relaxation intention (high alpha activity)
    relax_data = np.zeros((samples, 4))
    for ch in range(4):
        alpha_wave = np.sin(2 * np.pi * 10 * t) * 0.9  # High alpha for relaxation
        beta_wave = np.sin(2 * np.pi * 20 * t) * 0.2
        noise = np.random.normal(0, 0.1, samples)
        relax_data[:, ch] = alpha_wave + beta_wave + noise
    
    test_sessions.append({'data': relax_data, 'label': 'focus_decrease'})
    
    # Selection intention (gamma burst pattern)
    select_data = np.zeros((samples, 4))
    for ch in range(4):
        alpha_wave = np.sin(2 * np.pi * 10 * t) * 0.4
        beta_wave = np.sin(2 * np.pi * 20 * t) * 0.4
        gamma_burst = np.sin(2 * np.pi * 40 * t) * 0.6 * (t < 0.5)  # Gamma burst in first 500ms
        noise = np.random.normal(0, 0.1, samples)
        select_data[:, ch] = alpha_wave + beta_wave + gamma_burst + noise
    
    test_sessions.append({'data': select_data, 'label': 'select_object'})
    
    # Test preprocessing
    print("\n1. Testing advanced preprocessing...")
    test_data = focus_data
    
    # Apply filtering
    filtered_data = processor.apply_commercial_grade_filtering(test_data)
    print(f"   ✅ Commercial-grade filtering applied")
    print(f"   Original data range: {np.min(test_data):.3f} to {np.max(test_data):.3f}")
    print(f"   Filtered data range: {np.min(filtered_data):.3f} to {np.max(filtered_data):.3f}")
    
    # Remove artifacts
    clean_data = processor.remove_artifacts_advanced(filtered_data)
    print(f"   ✅ Advanced artifact removal completed")
    
    # Test feature extraction
    print("\n2. Testing advanced feature extraction...")
    features = processor.extract_advanced_features(clean_data)
    print(f"   ✅ Extracted {len(features)} advanced features")
    
    # Display some key features
    print("   Key features:")
    for i, (key, value) in enumerate(list(features.items())[:10]):
        print(f"     {key}: {value:.3f}")
    if len(features) > 10:
        print(f"     ... and {len(features) - 10} more features")
    
    # Test calibration
    print("\n3. Testing system calibration...")
    success = processor.calibrate_system(test_sessions)
    
    if success:
        print("   ✅ Calibration successful!")
        
        # Test intention detection
        print("\n4. Testing intention detection...")
        for i, session in enumerate(test_sessions):
            result = processor.detect_intentions(session['data'])
            
            print(f"   Test {i+1} (True label: {session['label']}):")
            print(f"     Status: {result['status']}")
            if result['status'] == 'success':
                print(f"     Predicted: {result['prediction']}")
                print(f"     Confidence: {result['confidence']:.3f}")
        
        # Save model
        model_path = '/Users/kunnath/Projects/brainwave/advanced_eeg_model.pkl'
        processor.save_model(model_path)
        print(f"\n5. Model saved to {model_path}")
        
    else:
        print("   ❌ Calibration failed")
    
    print("\n🎉 Advanced EEG processor test completed!")
