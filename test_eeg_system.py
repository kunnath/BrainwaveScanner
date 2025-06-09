"""
Simple EEG Test Script
Test the EEG collection and analysis system with minimal dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

# Import our modules
try:
    from eeg_collector import EEGDataCollector
    from brainwave_analyzer import BrainwaveAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required packages are installed")
    exit(1)

def run_eeg_test():
    """Run a comprehensive EEG system test"""
    
    print("🧠 EEG System Test Starting...")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1. Initializing EEG components...")
    collector = EEGDataCollector(headset_type="simulation", sample_rate=256)
    analyzer = BrainwaveAnalyzer(sample_rate=256)
    
    try:
        # 2. Start data collection
        print("\n2. Starting data collection...")
        collector.start_collection()
        
        # 3. Collect data for a few seconds
        print("3. Collecting EEG data (10 seconds)...")
        collected_data = []
        timestamps = []
        
        for i in range(10):
            time.sleep(1)
            data, ts = collector.get_latest_data(256)
            
            if len(data) > 0:
                collected_data.extend(data)
                timestamps.extend(ts)
                print(f"   Collected {len(data)} samples... Total: {len(collected_data)}")
            
        print(f"   ✅ Collection complete! Total samples: {len(collected_data)}")
        
        # 4. Convert to numpy array for analysis
        if len(collected_data) > 500:  # Need sufficient data for analysis
            eeg_data = np.array(collected_data)
            print(f"   Data shape: {eeg_data.shape}")
            
            # 5. Perform analysis
            print("\n4. Performing brainwave analysis...")
            
            # Basic features
            features = analyzer.extract_features(eeg_data)
            print("   ✅ Features extracted")
            
            # Band powers
            band_powers = analyzer.extract_band_power(eeg_data)
            print("   ✅ Band powers calculated")
            print("   Band Powers:")
            for band, power in band_powers.items():
                avg_power = np.mean(power)
                print(f"     {band.capitalize()}: {avg_power:.3f} µV²")
            
            # Mental states
            mental_states = analyzer.detect_mental_state(eeg_data)
            print("   ✅ Mental states detected")
            print("   Mental States:")
            for state, value in mental_states.items():
                print(f"     {state.capitalize()}: {value:.3f}")
            
            # 6. Create visualizations
            print("\n5. Creating analysis plots...")
            
            try:
                fig = analyzer.plot_analysis(eeg_data, "EEG Test Analysis")
                plt.savefig('/Users/kunnath/Projects/brainwave/eeg_test_results.png', 
                           dpi=300, bbox_inches='tight')
                print("   ✅ Analysis plots saved as 'eeg_test_results.png'")
                plt.close()
            except Exception as e:
                print(f"   ⚠️  Plot generation failed: {e}")
            
            # 7. Save results
            print("\n6. Saving test results...")
            
            results = {
                'test_info': {
                    'timestamp': datetime.now().isoformat(),
                    'duration_seconds': 10,
                    'total_samples': len(collected_data),
                    'sample_rate': 256,
                    'headset_type': 'simulation'
                },
                'band_powers': {k: float(np.mean(v)) for k, v in band_powers.items()},
                'mental_states': mental_states,
                'statistics': {
                    'mean_amplitude': float(np.mean(eeg_data)),
                    'std_amplitude': float(np.std(eeg_data)),
                    'max_amplitude': float(np.max(eeg_data)),
                    'min_amplitude': float(np.min(eeg_data))
                }
            }
            
            with open('/Users/kunnath/Projects/brainwave/test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print("   ✅ Results saved as 'test_results.json'")
            
            # 8. Summary
            print("\n" + "=" * 50)
            print("🎉 EEG SYSTEM TEST COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📊 Processed {len(collected_data)} EEG samples")
            print(f"🧠 Detected {len(mental_states)} mental state indicators")
            print(f"🌊 Analyzed {len(band_powers)} frequency bands")
            print("\n📈 Top Mental States:")
            sorted_states = sorted(mental_states.items(), key=lambda x: x[1], reverse=True)
            for state, value in sorted_states[:3]:
                print(f"   {state.capitalize()}: {value:.3f}")
            
            print("\n🌊 Dominant Frequency Bands:")
            sorted_bands = sorted(band_powers.items(), 
                                key=lambda x: np.mean(x[1]), reverse=True)
            for band, power in sorted_bands[:3]:
                print(f"   {band.capitalize()}: {np.mean(power):.3f} µV²")
            
            print("\n📁 Generated Files:")
            print("   - test_results.json (analysis data)")
            print("   - eeg_test_results.png (visualization)")
            
            return True
            
        else:
            print("   ❌ Insufficient data collected for analysis")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
        
    finally:
        # 9. Cleanup
        print("\n7. Cleaning up...")
        collector.cleanup()
        print("   ✅ Cleanup completed")

def run_realtime_demo():
    """Run a real-time demonstration"""
    
    print("\n🔴 REAL-TIME EEG DEMO")
    print("=" * 30)
    print("This will show live EEG analysis for 30 seconds...")
    print("Press Ctrl+C to stop early")
    
    collector = EEGDataCollector(headset_type="simulation", sample_rate=256)
    analyzer = BrainwaveAnalyzer(sample_rate=256)
    
    try:
        collector.start_collection()
        
        for i in range(30):  # Run for 30 seconds
            time.sleep(1)
            
            # Get latest data
            data, timestamps = collector.get_latest_data(512)  # 2 seconds of data
            
            if len(data) > 256:  # Need at least 1 second for analysis
                # Quick analysis
                mental_states = analyzer.detect_mental_state(data)
                
                # Display current state
                print(f"\r⏱️  {i+1:2d}s | ", end="")
                top_state = max(mental_states.items(), key=lambda x: x[1])
                print(f"🧠 {top_state[0].capitalize()}: {top_state[1]:.2f} | ", end="")
                
                # Show band activity
                band_powers = analyzer.extract_band_power(data)
                dominant_band = max(band_powers.items(), key=lambda x: np.mean(x[1]))
                print(f"🌊 {dominant_band[0].upper()}: {np.mean(dominant_band[1]):.1f}µV²", end="")
                
            else:
                print(f"\r⏱️  {i+1:2d}s | 📡 Collecting data...", end="")
        
        print("\n\n✅ Real-time demo completed!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo stopped by user")
        
    finally:
        collector.cleanup()

if __name__ == "__main__":
    print("🧠 EEG Brainwave Analysis System")
    print("================================")
    print()
    print("Choose an option:")
    print("1. Run comprehensive system test")
    print("2. Run real-time demo")
    print("3. Run both")
    print()
    
    try:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            success = run_eeg_test()
            if not success:
                print("\n⚠️  Test encountered issues. Check the output above.")
                
        elif choice == "2":
            run_realtime_demo()
            
        elif choice == "3":
            print("\n🔄 Running comprehensive test first...")
            success = run_eeg_test()
            
            if success:
                print("\n🔄 Now running real-time demo...")
                time.sleep(2)
                run_realtime_demo()
            else:
                print("\n⚠️  Skipping demo due to test failures")
                
        else:
            print("❌ Invalid choice. Please run the script again.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        
    print("\n🏁 Program finished.")
