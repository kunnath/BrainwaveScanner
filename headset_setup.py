"""
Headset Setup and Connection Guide
Instructions for connecting different EEG headsets
"""

import subprocess
import sys
import os
import time

class HeadsetSetup:
    """Guide for setting up different EEG headsets"""
    
    def __init__(self):
        self.headsets = {
            'muse': {
                'name': 'Muse 2/S Headset',
                'requirements': ['muselsl', 'pylsl', 'bluetooth'],
                'instructions': self.muse_setup,
                'price_range': '$200-300',
                'channels': '4-5',
                'best_for': 'Meditation, basic research'
            },
            'openbci': {
                'name': 'OpenBCI Cyton',
                'requirements': ['pyopenbci', 'pylsl', 'serial'],
                'instructions': self.openbci_setup,
                'price_range': '$500-1000',
                'channels': '8-16',
                'best_for': 'Research, full EEG mapping'
            },
            'neurosky': {
                'name': 'NeuroSky MindWave',
                'requirements': ['thinkgear', 'pyserial', 'bluetooth'],
                'instructions': self.neurosky_setup,
                'price_range': '$100-200',
                'channels': '1',
                'best_for': 'Simple BCI, demos'
            },
            'emotiv': {
                'name': 'Emotiv Insight/EPOC',
                'requirements': ['emotiv-sdk', 'pylsl'],
                'instructions': self.emotiv_setup,
                'price_range': '$300-800',
                'channels': '5-14',
                'best_for': 'Emotion detection, research'
            }
        }
    
    def print_headset_comparison(self):
        """Print comparison table of supported headsets"""
        print("🎧 SUPPORTED EEG HEADSETS COMPARISON")
        print("=" * 60)
        print(f"{'Headset':<20} {'Price':<12} {'Channels':<10} {'Best For':<15}")
        print("-" * 60)
        
        for key, info in self.headsets.items():
            print(f"{info['name']:<20} {info['price_range']:<12} {info['channels']:<10} {info['best_for']:<15}")
        
        print("\n" + "=" * 60)
    
    def muse_setup(self):
        """Setup instructions for Muse headset"""
        print("\n🎧 MUSE HEADSET SETUP")
        print("=" * 30)
        
        print("\n📋 Requirements:")
        print("- Muse 2 or Muse S headset")
        print("- Computer with Bluetooth")
        print("- muselsl package")
        
        print("\n🔧 Installation Steps:")
        print("1. Install muselsl:")
        print("   pip install muselsl")
        
        print("\n2. Pair your Muse headset:")
        print("   - Turn on Muse headset")
        print("   - Pair via Bluetooth settings")
        print("   - Note the device name (e.g., 'Muse-XXXX')")
        
        print("\n3. Start streaming:")
        print("   muselsl stream")
        print("   # Or specify device:")
        print("   muselsl stream --name 'Muse-XXXX'")
        
        print("\n4. Test connection:")
        print("   muselsl view")
        
        print("\n✅ Usage in Python:")
        print("   collector = EEGDataCollector(headset_type='muse')")
        
        print("\n💡 Tips:")
        print("- Ensure good electrode contact (wet slightly if needed)")
        print("- Minimize head movement during recording")
        print("- Use in quiet environment")
        
        return self.test_muse_connection()
    
    def openbci_setup(self):
        """Setup instructions for OpenBCI"""
        print("\n🎧 OPENBCI SETUP")
        print("=" * 20)
        
        print("\n📋 Requirements:")
        print("- OpenBCI Cyton board")
        print("- Ultracortex headset (or electrodes)")
        print("- USB dongle or Bluetooth")
        
        print("\n🔧 Installation Steps:")
        print("1. Install OpenBCI Python API:")
        print("   pip install pyopenbci")
        
        print("\n2. Connect hardware:")
        print("   - Attach electrodes to Ultracortex")
        print("   - Connect Cyton board")
        print("   - Plug in USB dongle")
        
        print("\n3. Find serial port:")
        print("   # On Mac/Linux:")
        print("   ls /dev/tty*")
        print("   # Look for /dev/ttyUSB0 or similar")
        
        print("\n4. Test connection:")
        print("   python -c \"from pyopenbci import OpenBCICyton; board = OpenBCICyton()\"")
        
        print("\n✅ Usage in Python:")
        print("   collector = EEGDataCollector(headset_type='openbci')")
        
        return self.test_openbci_connection()
    
    def neurosky_setup(self):
        """Setup instructions for NeuroSky"""
        print("\n🎧 NEUROSKY SETUP")
        print("=" * 22)
        
        print("\n📋 Requirements:")
        print("- NeuroSky MindWave Mobile 2")
        print("- Bluetooth connection")
        print("- ThinkGear library")
        
        print("\n🔧 Installation Steps:")
        print("1. Install dependencies:")
        print("   pip install pyserial")
        print("   pip install thinkgear")
        
        print("\n2. Pair headset:")
        print("   - Turn on MindWave")
        print("   - Pair via Bluetooth")
        print("   - Note connection port")
        
        print("\n3. Find Bluetooth port:")
        print("   # On Mac:")
        print("   ls /dev/cu.* | grep -i mindwave")
        print("   # On Windows: Check Device Manager")
        
        print("\n✅ Usage in Python:")
        print("   collector = EEGDataCollector(headset_type='neurosky')")
        
        return self.test_neurosky_connection()
    
    def emotiv_setup(self):
        """Setup instructions for Emotiv"""
        print("\n🎧 EMOTIV SETUP")
        print("=" * 18)
        
        print("\n📋 Requirements:")
        print("- Emotiv Insight or EPOC X")
        print("- Emotiv SDK license")
        print("- EmotivPRO software")
        
        print("\n🔧 Installation Steps:")
        print("1. Download EmotivPRO:")
        print("   - Visit emotiv.com")
        print("   - Download and install EmotivPRO")
        print("   - Purchase SDK license if needed")
        
        print("\n2. Install Python SDK:")
        print("   pip install cortex-python")
        
        print("\n3. Configure headset:")
        print("   - Wet electrodes with saline solution")
        print("   - Fit headset properly")
        print("   - Check signal quality in EmotivPRO")
        
        print("\n✅ Usage in Python:")
        print("   collector = EEGDataCollector(headset_type='emotiv')")
        
        print("\n⚠️  Note: Requires SDK subscription for commercial use")
        
        return self.test_emotiv_connection()
    
    def test_muse_connection(self):
        """Test Muse connection"""
        print("\n🔍 Testing Muse connection...")
        try:
            # Check if muselsl is installed
            import muselsl
            print("✅ muselsl package found")
            
            # Try to find streams
            try:
                from pylsl import resolve_stream
                streams = resolve_stream('type', 'EEG', timeout=2)
                if streams:
                    print("✅ Muse stream detected!")
                    return True
                else:
                    print("⚠️  No Muse stream found. Make sure:")
                    print("   - Headset is paired and on")
                    print("   - 'muselsl stream' is running")
                    return False
            except:
                print("⚠️  LSL not available for stream detection")
                return False
                
        except ImportError:
            print("❌ muselsl not installed. Run: pip install muselsl")
            return False
    
    def test_openbci_connection(self):
        """Test OpenBCI connection"""
        print("\n🔍 Testing OpenBCI connection...")
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            
            openbci_ports = [p for p in ports if 'OpenBCI' in str(p) or 'USB' in str(p)]
            
            if openbci_ports:
                print(f"✅ Potential OpenBCI port found: {openbci_ports[0]}")
                return True
            else:
                print("⚠️  No OpenBCI-like ports detected")
                print("   Available ports:")
                for port in ports:
                    print(f"   - {port}")
                return False
                
        except ImportError:
            print("❌ pyserial not installed. Run: pip install pyserial")
            return False
    
    def test_neurosky_connection(self):
        """Test NeuroSky connection"""
        print("\n🔍 Testing NeuroSky connection...")
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            
            mindwave_ports = [p for p in ports if 'mindwave' in str(p).lower() or 'hc-06' in str(p).lower()]
            
            if mindwave_ports:
                print(f"✅ MindWave port found: {mindwave_ports[0]}")
                return True
            else:
                print("⚠️  No MindWave ports detected")
                print("   Available Bluetooth ports:")
                bt_ports = [p for p in ports if 'bluetooth' in str(p).lower() or 'cu.' in str(p)]
                for port in bt_ports:
                    print(f"   - {port}")
                return False
                
        except ImportError:
            print("❌ pyserial not installed. Run: pip install pyserial")
            return False
    
    def test_emotiv_connection(self):
        """Test Emotiv connection"""
        print("\n🔍 Testing Emotiv connection...")
        print("⚠️  Emotiv requires EmotivPRO software and SDK license")
        print("   Check EmotivPRO for device status")
        return False
    
    def setup_headset(self, headset_type):
        """Run setup for specific headset"""
        if headset_type.lower() in self.headsets:
            info = self.headsets[headset_type.lower()]
            print(f"\n🎯 Setting up {info['name']}")
            return info['instructions']()
        else:
            print(f"❌ Unsupported headset type: {headset_type}")
            print("Supported types:", list(self.headsets.keys()))
            return False
    
    def install_requirements(self):
        """Install common requirements"""
        print("📦 Installing common requirements...")
        
        requirements = [
            'numpy',
            'scipy', 
            'matplotlib',
            'pandas',
            'pylsl',
            'pyserial',
            'muselsl'
        ]
        
        for req in requirements:
            try:
                print(f"Installing {req}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
                print(f"✅ {req} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install {req}")
    
    def interactive_setup(self):
        """Interactive setup wizard"""
        print("🧠 EEG HEADSET SETUP WIZARD")
        print("=" * 35)
        
        self.print_headset_comparison()
        
        print("\nWhich headset do you want to set up?")
        for i, (key, info) in enumerate(self.headsets.items(), 1):
            print(f"{i}. {info['name']}")
        print(f"{len(self.headsets)+1}. Install common requirements only")
        
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            
            if choice == len(self.headsets) + 1:
                self.install_requirements()
            elif 1 <= choice <= len(self.headsets):
                headset_key = list(self.headsets.keys())[choice-1]
                success = self.setup_headset(headset_key)
                
                if success:
                    print(f"\n🎉 {self.headsets[headset_key]['name']} setup completed!")
                else:
                    print(f"\n⚠️  {self.headsets[headset_key]['name']} setup needs attention")
            else:
                print("❌ Invalid choice")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Setup cancelled")

if __name__ == "__main__":
    setup = HeadsetSetup()
    setup.interactive_setup()
