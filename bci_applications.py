"""
Advanced BCI Applications using EEG Data
Demonstrates practical brain-computer interface applications
"""

import numpy as np
import time
import threading
import queue
from datetime import datetime
from typing import Dict, List, Callable
import json

# Simple audio feedback (works without pygame)
import os
import subprocess

try:
    from eeg_collector import EEGDataCollector
    from brainwave_analyzer import BrainwaveAnalyzer
except ImportError:
    print("Required modules not found. Make sure eeg_collector.py and brainwave_analyzer.py are available.")

class BCIController:
    """Brain-Computer Interface controller for various applications"""
    
    def __init__(self, headset_type: str = "simulation"):
        self.collector = EEGDataCollector(headset_type=headset_type, sample_rate=256)
        self.analyzer = BrainwaveAnalyzer(sample_rate=256)
        
        self.is_running = False
        self.applications = {}
        self.callbacks = {}
        
        # State tracking
        self.current_state = "neutral"
        self.state_history = []
        self.confidence_threshold = 0.6
        
    def register_application(self, name: str, callback: Callable):
        """Register a BCI application"""
        self.applications[name] = callback
        print(f"✅ Registered BCI application: {name}")
    
    def start_bci(self):
        """Start the BCI system"""
        self.collector.start_collection()
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("🧠 BCI system started!")
    
    def stop_bci(self):
        """Stop the BCI system"""
        self.is_running = False
        self.collector.stop_collection()
        print("🛑 BCI system stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get latest EEG data
                data, timestamps = self.collector.get_latest_data(512)  # 2 seconds
                
                if len(data) > 256:  # Need sufficient data
                    # Analyze mental state
                    mental_states = self.analyzer.detect_mental_state(data)
                    
                    # Determine dominant state
                    dominant_state = max(mental_states.items(), key=lambda x: x[1])
                    state_name, confidence = dominant_state
                    
                    # Update state if confidence is high enough
                    if confidence > self.confidence_threshold:
                        if state_name != self.current_state:
                            self._state_changed(state_name, confidence)
                    
                    # Record state history
                    self.state_history.append({
                        'timestamp': time.time(),
                        'state': state_name,
                        'confidence': confidence,
                        'all_states': mental_states
                    })
                    
                    # Keep only recent history
                    if len(self.state_history) > 100:
                        self.state_history = self.state_history[-100:]
                
                time.sleep(0.5)  # Process every 0.5 seconds
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(1)
    
    def _state_changed(self, new_state: str, confidence: float):
        """Handle state change"""
        old_state = self.current_state
        self.current_state = new_state
        
        print(f"🧠 State change: {old_state} → {new_state} (confidence: {confidence:.2f})")
        
        # Trigger applications
        for app_name, callback in self.applications.items():
            try:
                callback(new_state, confidence, old_state)
            except Exception as e:
                print(f"Application {app_name} error: {e}")
    
    def get_current_state(self):
        """Get current mental state"""
        return self.current_state
    
    def get_state_history(self, num_records: int = 10):
        """Get recent state history"""
        return self.state_history[-num_records:]

class BCIApplications:
    """Collection of practical BCI applications"""
    
    def __init__(self, bci_controller: BCIController):
        self.bci = bci_controller
        self.setup_applications()
    
    def setup_applications(self):
        """Setup all BCI applications"""
        # Register applications
        self.bci.register_application("meditation_feedback", self.meditation_feedback)
        self.bci.register_application("focus_trainer", self.focus_trainer)
        self.bci.register_application("stress_monitor", self.stress_monitor)
        self.bci.register_application("music_controller", self.music_controller)
        self.bci.register_application("light_controller", self.light_controller)
        
        # Application states
        self.meditation_session = {"active": False, "start_time": None, "duration": 0}
        self.focus_session = {"active": False, "focus_score": 0, "distractions": 0}
        self.stress_level = 0
        self.current_music = None
        self.light_state = {"brightness": 50, "color": "white"}
    
    def meditation_feedback(self, state: str, confidence: float, old_state: str):
        """Meditation guidance application"""
        if state == "meditation" and confidence > 0.7:
            if not self.meditation_session["active"]:
                self.meditation_session["active"] = True
                self.meditation_session["start_time"] = time.time()
                print("🧘‍♀️ Meditation session started")
                self._play_sound("meditation_start")
            
            # Update duration
            self.meditation_session["duration"] = time.time() - self.meditation_session["start_time"]
            
            # Provide feedback every 30 seconds
            if int(self.meditation_session["duration"]) % 30 == 0:
                print(f"🧘‍♀️ Great meditation! Duration: {self.meditation_session['duration']:.0f}s")
                self._play_sound("meditation_feedback")
        
        elif self.meditation_session["active"] and state != "meditation":
            print("🧘‍♀️ Meditation session paused - mind wandering detected")
            self._play_sound("meditation_reminder")
    
    def focus_trainer(self, state: str, confidence: float, old_state: str):
        """Focus training application"""
        if state == "concentration":
            if not self.focus_session["active"]:
                self.focus_session["active"] = True
                print("🎯 Focus training session started")
            
            # Increase focus score
            self.focus_session["focus_score"] += confidence * 10
            
            if self.focus_session["focus_score"] % 100 < 10:  # Every 100 points
                print(f"🎯 Focus score: {self.focus_session['focus_score']:.0f}")
                self._play_sound("focus_reward")
        
        elif self.focus_session["active"] and state != "concentration":
            self.focus_session["distractions"] += 1
            print(f"🎯 Focus lost - distraction #{self.focus_session['distractions']}")
            
            if state == "drowsiness":
                print("😴 Drowsiness detected - consider taking a break")
                self._play_sound("alertness_reminder")
    
    def stress_monitor(self, state: str, confidence: float, old_state: str):
        """Stress monitoring application"""
        # Simple stress level calculation
        if state == "alertness" and confidence > 0.8:
            self.stress_level = min(10, self.stress_level + 1)
        elif state == "relaxation":
            self.stress_level = max(0, self.stress_level - 1)
        
        # Stress alerts
        if self.stress_level > 7:
            print("⚠️  High stress detected - consider relaxation techniques")
            self._play_sound("stress_alert")
        elif self.stress_level < 3 and old_state != "relaxation":
            print("😌 Good relaxation level maintained")
    
    def music_controller(self, state: str, confidence: float, old_state: str):
        """Music selection based on mental state"""
        music_map = {
            "meditation": "ambient",
            "concentration": "focus_music", 
            "relaxation": "calm",
            "alertness": "energetic",
            "drowsiness": "upbeat"
        }
        
        if state in music_map and confidence > 0.6:
            new_music = music_map[state]
            if new_music != self.current_music:
                self.current_music = new_music
                print(f"🎵 Music changed to: {new_music}")
                self._control_music(new_music)
    
    def light_controller(self, state: str, confidence: float, old_state: str):
        """Smart lighting control based on mental state"""
        lighting_map = {
            "meditation": {"brightness": 20, "color": "warm_orange"},
            "concentration": {"brightness": 80, "color": "cool_white"},
            "relaxation": {"brightness": 40, "color": "soft_blue"},
            "alertness": {"brightness": 100, "color": "bright_white"},
            "drowsiness": {"brightness": 60, "color": "energizing_blue"}
        }
        
        if state in lighting_map and confidence > 0.6:
            new_settings = lighting_map[state]
            if new_settings != self.light_state:
                self.light_state = new_settings
                print(f"💡 Lights: {new_settings['brightness']}% {new_settings['color']}")
                self._control_lights(new_settings)
    
    def _play_sound(self, sound_type: str):
        """Play sound feedback (simplified)"""
        # In a real implementation, you would play actual audio files
        sound_messages = {
            "meditation_start": "🔔 Ding... (meditation bell)",
            "meditation_feedback": "🔔 Gentle chime",
            "meditation_reminder": "🔔 Soft reminder bell",
            "focus_reward": "✨ Success sound",
            "alertness_reminder": "⏰ Wake up chime", 
            "stress_alert": "⚠️  Attention tone"
        }
        
        if sound_type in sound_messages:
            print(f"🔊 {sound_messages[sound_type]}")
    
    def _control_music(self, music_type: str):
        """Control music player (placeholder)"""
        # In a real implementation, this would interface with Spotify, iTunes, etc.
        print(f"🎵 [Music Control] Playing {music_type} playlist")
    
    def _control_lights(self, settings: dict):
        """Control smart lights (placeholder)"""
        # In a real implementation, this would interface with Philips Hue, etc.
        print(f"💡 [Light Control] Set to {settings['brightness']}% {settings['color']}")
    
    def get_session_summary(self):
        """Get summary of all sessions"""
        summary = {
            "meditation": {
                "active": self.meditation_session["active"],
                "total_duration": self.meditation_session.get("duration", 0)
            },
            "focus": {
                "active": self.focus_session["active"],
                "score": self.focus_session["focus_score"],
                "distractions": self.focus_session["distractions"]
            },
            "stress_level": self.stress_level,
            "current_music": self.current_music,
            "light_state": self.light_state
        }
        return summary

def demo_bci_applications():
    """Run a demo of BCI applications"""
    print("🧠 BCI APPLICATIONS DEMO")
    print("=" * 30)
    
    # Initialize BCI system
    bci = BCIController(headset_type="simulation")
    apps = BCIApplications(bci)
    
    try:
        # Start BCI
        bci.start_bci()
        
        print("\n🚀 BCI applications are now running!")
        print("The system will automatically detect your mental states and trigger applications.")
        print("\nActive applications:")
        print("- 🧘‍♀️ Meditation feedback")
        print("- 🎯 Focus trainer") 
        print("- ⚠️  Stress monitor")
        print("- 🎵 Music controller")
        print("- 💡 Light controller")
        
        print("\nPress Ctrl+C to stop...")
        
        # Run for demo duration
        start_time = time.time()
        while time.time() - start_time < 60:  # Run for 1 minute
            time.sleep(5)
            
            # Show current status
            current_state = bci.get_current_state()
            summary = apps.get_session_summary()
            
            print(f"\n📊 Status Update:")
            print(f"   Current State: {current_state}")
            print(f"   Stress Level: {summary['stress_level']}/10")
            if summary['focus']['active']:
                print(f"   Focus Score: {summary['focus']['score']:.0f}")
            if summary['meditation']['active']:
                print(f"   Meditation: {summary['meditation']['total_duration']:.0f}s")
        
        print("\n✅ Demo completed!")
        
        # Final summary
        final_summary = apps.get_session_summary()
        print("\n📈 FINAL SUMMARY:")
        print(f"   Total meditation time: {final_summary['meditation']['total_duration']:.0f}s")
        print(f"   Focus score achieved: {final_summary['focus']['score']:.0f}")
        print(f"   Distractions: {final_summary['focus']['distractions']}")
        print(f"   Final stress level: {final_summary['stress_level']}/10")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    
    finally:
        bci.stop_bci()

def create_custom_application():
    """Create a custom BCI application"""
    print("\n🛠️ CUSTOM BCI APPLICATION BUILDER")
    print("=" * 40)
    
    print("This example shows how to create your own BCI application:")
    
    code_example = '''
def my_custom_app(state, confidence, old_state):
    """Custom BCI application example"""
    
    if state == "concentration" and confidence > 0.8:
        print("🎮 High concentration detected - boost game performance!")
        # Could control game difficulty, provide rewards, etc.
    
    elif state == "relaxation" and confidence > 0.7:
        print("🌅 Relaxed state - perfect for creative work!")
        # Could switch to creative apps, dim lights, play ambient sounds
    
    elif state == "meditation" and confidence > 0.6:
        print("🧘‍♀️ Meditation detected - entering zen mode!")
        # Could activate meditation timer, breathing guides, etc.

# Register your custom application
bci.register_application("my_custom_app", my_custom_app)
'''
    
    print(code_example)
    
    print("\n💡 BCI Application Ideas:")
    print("- 🎮 Game difficulty adjustment")
    print("- 📚 Study session optimization") 
    print("- 🏠 Smart home automation")
    print("- 🎨 Creative workflow enhancement")
    print("- 🚗 Driver alertness monitoring")
    print("- 💊 Medication reminders")
    print("- 🏋️‍♀️ Workout intensity control")
    print("- 📱 App switching based on mental state")

if __name__ == "__main__":
    print("🧠 BRAIN-COMPUTER INTERFACE APPLICATIONS")
    print("=" * 45)
    
    print("\nChoose an option:")
    print("1. Run BCI applications demo")
    print("2. Show custom application examples")
    print("3. Both")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            demo_bci_applications()
        elif choice == "2":
            create_custom_application()
        elif choice == "3":
            demo_bci_applications()
            create_custom_application()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
