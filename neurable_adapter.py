"""
Regular Headset to Neurable-style Adapter
Converts regular EEG headset data to work with advanced processing
"""

import numpy as np
import pandas as pd
from datetime import datetime
import threading
import queue
import time
from advanced_eeg_processor import AdvancedEEGProcessor
from eeg_collector import EEGDataCollector

class NeurableStyleAdapter:
    """
    Adapter to make regular EEG headsets work with Neurable-inspired techniques
    """
    
    def __init__(self, headset_type="simulation", sample_rate=256):
        self.headset_type = headset_type
        self.sample_rate = sample_rate
        
        # Initialize components
        self.eeg_collector = EEGDataCollector(headset_type=headset_type, sample_rate=sample_rate)
        self.advanced_processor = AdvancedEEGProcessor(sample_rate=sample_rate)
        
        # Calibration settings
        self.calibration_complete = False
        self.calibration_data = []
        
        # Real-time processing
        self.is_running = False
        self.processing_thread = None
        self.results_queue = queue.Queue()
        
        # Intent mapping for regular headsets
        self.headset_intent_mapping = {
            'muse': {
                'electrode_positions': ['TP9', 'AF7', 'AF8', 'TP10'],
                'focus_channels': [1, 2],  # AF7, AF8 for frontal focus
                'relaxation_channels': [0, 3],  # TP9, TP10 for temporal relaxation
            },
            'emotiv': {
                'electrode_positions': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
                'focus_channels': [2, 11],  # F3, F4 for focus
                'relaxation_channels': [6, 7],  # O1, O2 for relaxation
            },
            'openbci': {
                'electrode_positions': ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2'],
                'focus_channels': [0, 1],  # Fp1, Fp2 for focus
                'relaxation_channels': [6, 7],  # O1, O2 for relaxation
            },
            'simulation': {
                'electrode_positions': ['Fp1', 'Fp2', 'C3', 'C4'],
                'focus_channels': [0, 1],
                'relaxation_channels': [2, 3],
            }
        }
    
    def calibrate_for_regular_headset(self, headset_specific_type="simulation"):
        """
        Perform calibration optimized for regular headset capabilities
        """
        print(f"🎯 Starting calibration for {headset_specific_type} headset...")
        print("This will simulate the Neurable calibration process")
        
        # Get headset configuration
        config = self.headset_intent_mapping.get(headset_specific_type, 
                                                self.headset_intent_mapping['simulation'])
        
        print(f"Electrode positions: {config['electrode_positions']}")
        print(f"Focus channels: {config['focus_channels']}")
        print(f"Relaxation channels: {config['relaxation_channels']}")
        
        # Start data collection for calibration
        self.eeg_collector.start_collection()
        
        calibration_sessions = []
        
        # Calibration session 1: Baseline (eyes closed, relaxed)
        print("\n📊 Calibration Session 1: Baseline")
        print("Please close your eyes and relax for 10 seconds...")
        self._countdown(3)
        
        baseline_data = self._collect_calibration_data(duration=10, session_name="baseline")
        calibration_sessions.append({
            'data': baseline_data,
            'label': 'baseline',
            'session_type': 'baseline'
        })
        
        # Calibration session 2: Focus task
        print("\n🎯 Calibration Session 2: Focus Task")
        print("Please focus intensely on a mental math problem for 10 seconds...")
        print("Example: Calculate 17 × 23 in your head")
        self._countdown(3)
        
        focus_data = self._collect_calibration_data(duration=10, session_name="focus")
        calibration_sessions.append({
            'data': focus_data,
            'label': 'focus_increase',
            'session_type': 'focus'
        })
        
        # Calibration session 3: Relaxation
        print("\n😌 Calibration Session 3: Relaxation")
        print("Please think of a peaceful, calm scene for 10 seconds...")
        self._countdown(3)
        
        relax_data = self._collect_calibration_data(duration=10, session_name="relaxation")
        calibration_sessions.append({
            'data': relax_data,
            'label': 'focus_decrease',
            'session_type': 'relaxation'
        })
        
        # Calibration session 4: Selection intent
        print("\n👆 Calibration Session 4: Selection Intent")
        print("Imagine clicking or selecting something repeatedly for 10 seconds...")
        self._countdown(3)
        
        select_data = self._collect_calibration_data(duration=10, session_name="selection")
        calibration_sessions.append({
            'data': select_data,
            'label': 'select_object',
            'session_type': 'selection'
        })
        
        # Stop data collection
        self.eeg_collector.cleanup()
        
        # Process calibration data
        print("\n🔄 Processing calibration data...")
        processed_sessions = []
        
        for session in calibration_sessions:
            # Apply headset-specific processing
            processed_data = self._process_for_headset_type(session['data'], headset_specific_type)
            
            processed_sessions.append({
                'data': processed_data,
                'label': session['label']
            })
        
        # Train the advanced processor
        print("🧠 Training advanced neural classifier...")
        success = self.advanced_processor.calibrate_system(processed_sessions)
        
        if success:
            self.calibration_complete = True
            print("✅ Calibration completed successfully!")
            
            # Save calibration data
            self._save_calibration_data(calibration_sessions, headset_specific_type)
            
            return True
        else:
            print("❌ Calibration failed")
            return False
    
    def _process_for_headset_type(self, data, headset_type):
        """
        Apply headset-specific processing to optimize for regular headsets
        """
        # Apply advanced filtering
        filtered_data = self.advanced_processor.apply_commercial_grade_filtering(data)
        
        # Apply advanced artifact removal
        clean_data = self.advanced_processor.remove_artifacts_advanced(filtered_data)
        
        # Headset-specific optimization
        config = self.headset_intent_mapping.get(headset_type, 
                                                self.headset_intent_mapping['simulation'])
        
        # Focus on specific channels for this headset type
        if clean_data.shape[1] >= max(config['focus_channels'] + config['relaxation_channels']):
            # Weight important channels more heavily
            processed_data = clean_data.copy()
            
            # Enhance focus-related channels
            for ch in config['focus_channels']:
                if ch < processed_data.shape[1]:
                    processed_data[:, ch] *= 1.2  # Slight amplification
            
            # Enhance relaxation-related channels  
            for ch in config['relaxation_channels']:
                if ch < processed_data.shape[1]:
                    processed_data[:, ch] *= 1.1  # Slight amplification
        else:
            processed_data = clean_data
        
        return processed_data
    
    def start_real_time_processing(self, headset_specific_type="simulation"):
        """
        Start real-time intention detection
        """
        if not self.calibration_complete:
            print("❌ Please complete calibration first")
            return False
        
        print("🔴 Starting real-time intention detection...")
        
        self.eeg_collector.start_collection()
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._real_time_processing_loop,
            args=(headset_specific_type,)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True
    
    def stop_real_time_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        self.eeg_collector.cleanup()
        print("⏹️ Real-time processing stopped")
    
    def _real_time_processing_loop(self, headset_type):
        """Real-time processing loop"""
        window_size = int(2 * self.sample_rate)  # 2-second windows
        
        while self.is_running:
            try:
                # Get latest data
                data, timestamps = self.eeg_collector.get_latest_data(window_size)
                
                if len(data) >= window_size:
                    # Process data
                    processed_data = self._process_for_headset_type(np.array(data), headset_type)
                    
                    # Detect intentions
                    result = self.advanced_processor.detect_intentions(processed_data)
                    
                    # Add timestamp and put in results queue
                    result['timestamp'] = datetime.now()
                    result['processed_data_shape'] = processed_data.shape
                    
                    self.results_queue.put(result)
                
                time.sleep(0.5)  # Process every 500ms
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def get_latest_intention(self):
        """Get the latest intention detection result"""
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_pending_intentions(self):
        """Get all pending intention results"""
        results = []
        
        while True:
            try:
                result = self.results_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def _collect_calibration_data(self, duration, session_name):
        """Collect data for calibration session"""
        collected_data = []
        start_time = time.time()
        
        print(f"Recording {session_name} data...")
        
        while time.time() - start_time < duration:
            data, timestamps = self.eeg_collector.get_latest_data(256)  # 1 second
            if len(data) > 0:
                collected_data.extend(data)
            
            # Show progress
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            print(f"\rProgress: {elapsed:.1f}s / {duration}s (remaining: {remaining:.1f}s)", end="")
            
            time.sleep(0.1)
        
        print(f"\n✅ Collected {len(collected_data)} samples for {session_name}")
        return np.array(collected_data)
    
    def _countdown(self, seconds):
        """Countdown timer"""
        for i in range(seconds, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
        print("START!")
    
    def _save_calibration_data(self, calibration_sessions, headset_type):
        """Save calibration data for future use"""
        calibration_info = {
            'timestamp': datetime.now().isoformat(),
            'headset_type': headset_type,
            'sample_rate': self.sample_rate,
            'sessions': []
        }
        
        for session in calibration_sessions:
            session_info = {
                'label': session['label'],
                'session_type': session['session_type'],
                'data_shape': session['data'].shape,
                'data_stats': {
                    'mean': float(np.mean(session['data'])),
                    'std': float(np.std(session['data'])),
                    'min': float(np.min(session['data'])),
                    'max': float(np.max(session['data']))
                }
            }
            calibration_info['sessions'].append(session_info)
        
        # Save calibration info
        import json
        with open('/Users/kunnath/Projects/brainwave/calibration_info.json', 'w') as f:
            json.dump(calibration_info, f, indent=2)
        
        print("📁 Calibration info saved to calibration_info.json")
    
    def demonstrate_capabilities(self, headset_type="simulation"):
        """
        Demonstrate the Neurable-style capabilities with a regular headset
        """
        print("🎮 Neurable-Style EEG Demonstration")
        print("=" * 50)
        
        print(f"\nUsing {headset_type} headset configuration")
        config = self.headset_intent_mapping.get(headset_type, 
                                                self.headset_intent_mapping['simulation'])
        
        print(f"📍 Electrode positions: {config['electrode_positions']}")
        print(f"🎯 Focus detection channels: {[config['electrode_positions'][i] for i in config['focus_channels']]}")
        print(f"😌 Relaxation detection channels: {[config['electrode_positions'][i] for i in config['relaxation_channels']]}")
        
        # Perform calibration
        print("\n🎯 Step 1: Calibration")
        success = self.calibrate_for_regular_headset(headset_type)
        
        if not success:
            print("❌ Calibration failed. Cannot proceed with demonstration.")
            return
        
        # Start real-time demonstration
        print("\n🔴 Step 2: Real-time Intention Detection")
        self.start_real_time_processing(headset_type)
        
        # Demonstrate for 30 seconds
        print("Running live demonstration for 30 seconds...")
        print("Try different mental states:")
        print("  - Focus intensely (mental math)")
        print("  - Relax and think peaceful thoughts")
        print("  - Imagine clicking/selecting objects")
        
        demo_start = time.time()
        intention_history = []
        
        while time.time() - demo_start < 30:
            # Get latest intention
            intention = self.get_latest_intention()
            
            if intention and intention['status'] == 'success':
                timestamp = intention['timestamp']
                prediction = intention['prediction']
                confidence = intention['confidence']
                
                intention_history.append({
                    'time': timestamp,
                    'intention': prediction,
                    'confidence': confidence
                })
                
                # Display current intention
                elapsed = time.time() - demo_start
                print(f"\r⏱️  {elapsed:5.1f}s | 🧠 {prediction:15s} | 📊 {confidence:5.2f}", end="")
            
            time.sleep(0.1)
        
        # Stop processing
        self.stop_real_time_processing()
        
        # Show summary
        print("\n\n📊 Demonstration Summary")
        print("=" * 30)
        
        if intention_history:
            # Count intentions
            intention_counts = {}
            confidence_by_intention = {}
            
            for record in intention_history:
                intention = record['intention']
                confidence = record['confidence']
                
                if intention not in intention_counts:
                    intention_counts[intention] = 0
                    confidence_by_intention[intention] = []
                
                intention_counts[intention] += 1
                confidence_by_intention[intention].append(confidence)
            
            print(f"Total detections: {len(intention_history)}")
            print("\nIntention breakdown:")
            
            for intention, count in sorted(intention_counts.items(), key=lambda x: x[1], reverse=True):
                avg_confidence = np.mean(confidence_by_intention[intention])
                percentage = (count / len(intention_history)) * 100
                print(f"  {intention:15s}: {count:3d} times ({percentage:5.1f}%) - Avg confidence: {avg_confidence:.3f}")
            
            # Show timeline of last 10 detections
            print(f"\nLast 10 detections:")
            for record in intention_history[-10:]:
                time_str = record['time'].strftime("%H:%M:%S")
                print(f"  {time_str} | {record['intention']:15s} | {record['confidence']:.3f}")
        
        else:
            print("No intentions detected during demonstration")
        
        print("\n✅ Demonstration completed!")


def main():
    """Main demonstration function"""
    print("🧠 Regular Headset → Neurable-Style Adapter")
    print("=" * 50)
    
    # Available headset types
    headset_types = ['muse', 'emotiv', 'openbci', 'simulation']
    
    print("Available headset types:")
    for i, headset in enumerate(headset_types, 1):
        print(f"  {i}. {headset}")
    
    # For demo, use simulation
    print(f"\nUsing simulation headset for demonstration...")
    
    # Create adapter
    adapter = NeurableStyleAdapter(headset_type="simulation", sample_rate=256)
    
    # Run demonstration
    adapter.demonstrate_capabilities("simulation")


if __name__ == "__main__":
    main()
