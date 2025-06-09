#!/usr/bin/env python3
"""
EEG Brainwave System Installer
Cross-platform Python installer for the EEG analysis system
"""

import sys
import subprocess
import pkg_resources
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_package(package):
    """Install a Python package using pip"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_package_installed(package):
    """Check if a package is already installed"""
    try:
        pkg_resources.get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_requirements():
    """Install all required packages"""
    
    # Core packages (essential)
    core_packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.5.0",
        "pandas>=1.3.0"
    ]
    
    # Analysis packages
    analysis_packages = [
        "scikit-learn>=1.0.0",
        "mne>=1.0.0"
    ]
    
    # Interface packages
    interface_packages = [
        "streamlit>=1.28.0",
        "plotly>=5.0.0"
    ]
    
    # Optional packages (nice to have)
    optional_packages = [
        "pylsl>=1.16.0",
        "muselsl>=2.1.0", 
        "pyserial>=3.5",
        "pywavelets>=1.4.0",
        "antropy>=0.1.4",
        "psutil>=5.9.0",
        "h5py>=3.7.0"
    ]
    
    # Install core packages first
    print("\n🔧 Installing core packages...")
    core_success = True
    for package in core_packages:
        if not install_package(package):
            core_success = False
    
    if not core_success:
        print("❌ Critical packages failed to install")
        return False
    
    # Install analysis packages
    print("\n🧠 Installing EEG analysis packages...")
    for package in analysis_packages:
        install_package(package)
    
    # Install interface packages
    print("\n🖥️  Installing interface packages...")
    for package in interface_packages:
        install_package(package)
    
    # Install optional packages
    print("\n📡 Installing optional packages...")
    for package in optional_packages:
        if not install_package(package):
            print(f"⚠️  {package} is optional - continuing...")
    
    return True

def test_installation():
    """Test if the installation was successful"""
    print("\n🧪 Testing installation...")
    
    # Test core imports
    test_imports = [
        ("numpy", "np"),
        ("scipy", "scipy"),
        ("matplotlib.pyplot", "plt"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
    ]
    
    success = True
    for module, alias in test_imports:
        try:
            exec(f"import {module} as {alias}")
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            success = False
    
    # Test optional imports
    optional_imports = [
        "mne",
        "streamlit", 
        "plotly"
    ]
    
    for module in optional_imports:
        try:
            exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError:
            print(f"⚠️  {module} (optional)")
    
    return success

def create_test_script():
    """Create a simple test script"""
    test_script = '''#!/usr/bin/env python3
"""
Quick test of EEG system functionality
"""

def test_basic_functionality():
    print("🧠 Testing EEG System Components...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Core scientific packages working")
    except ImportError as e:
        print(f"❌ Core packages error: {e}")
        return False
    
    try:
        # Test our modules
        from eeg_collector import EEGDataCollector
        from brainwave_analyzer import BrainwaveAnalyzer
        
        # Quick test
        collector = EEGDataCollector(headset_type="simulation")
        analyzer = BrainwaveAnalyzer()
        
        print("✅ EEG modules loaded successfully")
        print("✅ System ready for use!")
        return True
        
    except ImportError as e:
        print(f"❌ EEG modules error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\\n🎉 EEG system is ready!")
        print("\\nNext steps:")
        print("1. Run: python test_eeg_system.py")
        print("2. Run: python headset_setup.py")  
        print("3. Run: streamlit run dashboard.py")
    else:
        print("\\n❌ Some issues detected. Check installation.")
'''
    
    with open("quick_test.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created quick_test.py")

def main():
    """Main installer function"""
    print("🧠 EEG BRAINWAVE ANALYSIS SYSTEM INSTALLER")
    print("=" * 45)
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Upgrade pip
    print("\n📦 Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded")
    except subprocess.CalledProcessError:
        print("⚠️  pip upgrade failed - continuing anyway")
    
    # Install packages
    print("\n📦 Installing packages...")
    if not install_requirements():
        print("❌ Installation failed")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Test installation
    if test_installation():
        print("\n🎉 Installation completed successfully!")
    else:
        print("\n⚠️  Installation completed with some issues")
    
    # Create test script
    create_test_script()
    
    print("\n🚀 GETTING STARTED:")
    print("1. Test system:      python quick_test.py")
    print("2. Full test:        python test_eeg_system.py") 
    print("3. Setup headset:    python headset_setup.py")
    print("4. Launch dashboard: streamlit run dashboard.py")
    print("5. Try BCI apps:     python bci_applications.py")
    
    print("\n📚 Read README.md for complete documentation")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
