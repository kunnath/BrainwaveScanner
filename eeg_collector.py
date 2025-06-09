"""
Advanced EEG Data Collector
Supports multiple headset types with real-time streaming and analysis
"""

import numpy as np
import pandas as pd
import time
import threading
import queue
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Signal processing
from scipy import signal
from scipy.fft import fft, fftfreq
import mne

# LSL for real-time streaming
try:
    from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    print("pylsl not available - some streaming features disabled")

# Headset-specific imports
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class EEGDataCollector:
    """
    Universal EEG data collector supporting multiple headset types
    """
    
    def __init__(self, headset_type: str = "simulation", sample_rate: int = 256):
        self.headset_type = headset_type.lower()
        self.sample_rate = sample_rate
        self.is_collecting = False
        self.data_queue = queue.Queue()
        self.raw_data = []
        self.timestamps = []
        self.collection_thread = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize headset connection
        self.inlet = None
        self.serial_connection = None
        self._setup_headset()
        
    def _setup_headset(self):
        """Setup connection based on headset type"""
        if self.headset_type == "muse":
            self._setup_muse()
        elif self.headset_type == "openbci":
            self._setup_openbci()
        elif self.headset_type == "neurosky":
            self._setup_neurosky()
        elif self.headset_type == "emotiv":
            self._setup_emotiv()
        else:
            self.logger.info("Using simulation mode")
            
    def _setup_muse(self):
        """Setup Muse headset connection via LSL"""
        if not LSL_AVAILABLE:
            self.logger.error("LSL not available for Muse connection")
            return
            
        try:
            self.logger.info("Looking for Muse EEG stream...")
            streams = resolve_stream('type', 'EEG')
            if streams:
                self.inlet = StreamInlet(streams[0])
                self.logger.info("Muse headset connected!")
            else:
                self.logger.warning("No Muse stream found. Make sure muselsl is streaming.")
        except Exception as e:
            self.logger.error(f"Muse setup failed: {e}")
            
    def _setup_openbci(self):
        """Setup OpenBCI connection"""
        try:
            # OpenBCI typically connects via serial or LSL
            if LSL_AVAILABLE:
                streams = resolve_stream('name', 'OpenBCIEEG')
                if streams:
                    self.inlet = StreamInlet(streams[0])
                    self.logger.info("OpenBCI connected via LSL!")
                else:
                    self.logger.warning("OpenBCI LSL stream not found")
        except Exception as e:
            self.logger.error(f"OpenBCI setup failed: {e}")
            
    def _setup_neurosky(self):
        """Setup NeuroSky connection"""
        if not SERIAL_AVAILABLE:
            self.logger.error("Serial not available for NeuroSky")
            return
            
        # NeuroSky typically uses serial or bluetooth
        try:
            # Common NeuroSky ports
            ports = ['/dev/tty.MindWaveMobile-SerialPo', '/dev/cu.HC-06-DevB', 'COM3', 'COM4']
            for port in ports:
                try:
                    self.serial_connection = serial.Serial(port, 57600, timeout=1)
                    self.logger.info(f"NeuroSky connected on {port}")
                    break
                except:
                    continue
        except Exception as e:
            self.logger.error(f"NeuroSky setup failed: {e}")
            
    def _setup_emotiv(self):
        """Setup Emotiv connection"""
        try:
            # Emotiv typically uses their SDK or LSL
            if LSL_AVAILABLE:
                streams = resolve_stream('name', 'EmotivDataStream-EEG')
                if streams:
                    self.inlet = StreamInlet(streams[0])
                    self.logger.info("Emotiv connected!")
        except Exception as e:
            self.logger.error(f"Emotiv setup failed: {e}")
    
    def _generate_simulation_data(self) -> Tuple[List[float], float]:
        """Generate realistic EEG simulation data"""
        timestamp = time.time()
        
        # Simulate 4-channel EEG with realistic brainwave patterns
        channels = 4
        sample_duration = 1.0 / self.sample_rate
        
        # Generate different brainwave frequencies
        alpha = 10  # 8-13 Hz
        beta = 20   # 13-30 Hz
        theta = 6   # 4-8 Hz
        delta = 3   # 0.5-4 Hz
        
        t = timestamp % 1.0  # Use fractional part for continuous wave
        
        # Mix different brainwave patterns for each channel
        eeg_sample = []
        for ch in range(channels):
            # Each channel has different dominant frequencies
            if ch == 0:  # Frontal - more beta activity
                signal_val = (np.sin(2 * np.pi * beta * t) * 0.5 +
                             np.sin(2 * np.pi * alpha * t) * 0.3 +
                             np.random.normal(0, 0.1))
            elif ch == 1:  # Parietal - more alpha
                signal_val = (np.sin(2 * np.pi * alpha * t) * 0.6 +
                             np.sin(2 * np.pi * theta * t) * 0.2 +
                             np.random.normal(0, 0.1))
            elif ch == 2:  # Temporal - mixed
                signal_val = (np.sin(2 * np.pi * alpha * t) * 0.4 +
                             np.sin(2 * np.pi * beta * t) * 0.3 +
                             np.sin(2 * np.pi * theta * t) * 0.2 +
                             np.random.normal(0, 0.1))
            else:  # Occipital - more alpha and theta
                signal_val = (np.sin(2 * np.pi * alpha * t) * 0.5 +
                             np.sin(2 * np.pi * theta * t) * 0.3 +
                             np.random.normal(0, 0.1))
                             
            # Scale to microvolts (typical EEG range: -100 to 100 µV)
            eeg_sample.append(signal_val * 50)
            
        return eeg_sample, timestamp
    
    def _collect_data_thread(self):
        """Background thread for data collection"""
        self.logger.info("Starting EEG data collection...")
        
        while self.is_collecting:
            try:
                if self.inlet and LSL_AVAILABLE:
                    # Real headset data via LSL
                    sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                    if sample:
                        self.data_queue.put((sample, timestamp))
                        
                elif self.serial_connection:
                    # NeuroSky serial data
                    data = self.serial_connection.readline()
                    if data:
                        # Parse NeuroSky data format
                        timestamp = time.time()
                        # Simplified parsing - adapt based on actual format
                        sample = [float(data.strip())]
                        self.data_queue.put((sample, timestamp))
                        
                else:
                    # Simulation mode
                    sample, timestamp = self._generate_simulation_data()
                    self.data_queue.put((sample, timestamp))
                    
                # Control sampling rate
                time.sleep(1.0 / self.sample_rate)
                
            except Exception as e:
                self.logger.error(f"Data collection error: {e}")
                time.sleep(0.1)
                
    def start_collection(self):
        """Start EEG data collection"""
        if self.is_collecting:
            self.logger.warning("Collection already running")
            return
            
        self.is_collecting = True
        self.raw_data = []
        self.timestamps = []
        
        self.collection_thread = threading.Thread(target=self._collect_data_thread)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        self.logger.info("EEG collection started")
        
    def stop_collection(self):
        """Stop EEG data collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2)
        self.logger.info("EEG collection stopped")
        
    def get_latest_data(self, num_samples: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Get the latest EEG samples"""
        # Collect data from queue
        while not self.data_queue.empty():
            try:
                sample, timestamp = self.data_queue.get_nowait()
                self.raw_data.append(sample)
                self.timestamps.append(timestamp)
            except queue.Empty:
                break
                
        # Keep only recent data
        if len(self.raw_data) > num_samples * 2:
            self.raw_data = self.raw_data[-num_samples * 2:]
            self.timestamps = self.timestamps[-num_samples * 2:]
            
        # Return latest samples
        if len(self.raw_data) >= num_samples:
            recent_data = np.array(self.raw_data[-num_samples:])
            recent_timestamps = np.array(self.timestamps[-num_samples:])
            return recent_data, recent_timestamps
        else:
            return np.array(self.raw_data), np.array(self.timestamps)
            
    def save_data(self, filename: str):
        """Save collected data to file"""
        if not self.raw_data:
            self.logger.warning("No data to save")
            return
            
        data_dict = {
            'eeg_data': self.raw_data,
            'timestamps': self.timestamps,
            'sample_rate': self.sample_rate,
            'headset_type': self.headset_type,
            'collection_date': datetime.now().isoformat()
        }
        
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(data_dict, f, indent=2, default=str)
        else:
            # Save as CSV
            df = pd.DataFrame(self.raw_data)
            df['timestamp'] = self.timestamps
            df.to_csv(filename, index=False)
            
        self.logger.info(f"Data saved to {filename}")
        
    def cleanup(self):
        """Clean up connections"""
        self.stop_collection()
        if self.serial_connection:
            self.serial_connection.close()
        self.logger.info("Cleanup completed")


if __name__ == "__main__":
    # Test the collector
    print("🧠 EEG Data Collector Test")
    print("=" * 40)
    
    # Initialize collector (simulation mode for testing)
    collector = EEGDataCollector(headset_type="simulation", sample_rate=256)
    
    try:
        # Start collection
        collector.start_collection()
        print("Collection started... Press Ctrl+C to stop")
        
        # Collect for 10 seconds
        for i in range(10):
            time.sleep(1)
            data, timestamps = collector.get_latest_data(256)
            if len(data) > 0:
                print(f"Collected {len(data)} samples, latest: {data[-1][:2]}...")
                
        # Save data
        collector.save_data("test_eeg_data.csv")
        
    except KeyboardInterrupt:
        print("\nStopping collection...")
    finally:
        collector.cleanup()
        print("Test completed!")
