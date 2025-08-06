#!/usr/bin/env python
"""Demo script showing TTS integration with a simulated RAG response."""

import time
import subprocess
import platform
from pathlib import Path
from services import TTSService, EmotionalTTSService
from utils import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def play_audio(audio_path):
    """Play audio file using system's default audio player."""
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", str(audio_path)], check=False)
        elif system == "Linux":
            for player in ["aplay", "paplay", "ffplay"]:
                try:
                    subprocess.run([player, str(audio_path)], check=False)
                    break
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            subprocess.run(["start", "", str(audio_path)], shell=True, check=False)
    except Exception as e:
        logger.warning(f"Failed to play audio: {e}")


def simulate_rag_response(question):
    """Simulate a RAG response for demonstration purposes."""
    responses = {
        "motor learning": (
            "Motor learning refers to the process of acquiring and refining motor skills through practice and experience. "
            "According to the literature, it involves changes in the central nervous system that lead to relatively permanent "
            "improvements in performance. The cerebellum plays a crucial role in motor learning, particularly in adaptation "
            "and error correction. Studies show that motor learning occurs on multiple timescales, with fast and slow "
            "adaptive processes working together to optimize movement patterns."
        ),
        "gait adaptation": (
            "Gait adaptation is the process by which humans modify their walking patterns in response to environmental "
            "changes or perturbations. Research demonstrates that gait adaptation operates on dual timescales, with initial "
            "rapid adjustments followed by slower consolidation. The nervous system uses sensory feedback to continuously "
            "update internal models of locomotion. This adaptive capability is essential for navigating complex terrains "
            "and recovering from disturbances during walking."
        ),
        "biomechanics": (
            "Biomechanics is the study of mechanical laws relating to the movement and structure of living organisms. "
            "In human movement, biomechanics examines forces, torques, and energy transfer during activities like walking, "
            "running, and balance. Key concepts include center of mass dynamics, joint moments, and muscle-tendon mechanics. "
            "Understanding biomechanics is crucial for optimizing performance, preventing injuries, and designing assistive "
            "devices like exoskeletons."
        ),
        "default": (
            "Based on the scientific literature, this topic involves complex interactions between neural control systems "
            "and mechanical properties of the musculoskeletal system. Research indicates that multiple control mechanisms "
            "work in parallel to ensure robust and adaptable movement. The integration of sensory feedback with predictive "
            "control allows for both stability and flexibility in human movement patterns."
        )
    }
    
    # Simple keyword matching
    question_lower = question.lower()
    if "motor learning" in question_lower:
        return responses["motor learning"]
    elif "gait" in question_lower or "adaptation" in question_lower:
        return responses["gait adaptation"]
    elif "biomechanics" in question_lower:
        return responses["biomechanics"]
    else:
        return responses["default"]


def main():
    """Main demo function."""
    print("=" * 60)
    print("🤖 RAG System with Chatterbox TTS Demo")
    print("=" * 60)
    
    # Load configuration
    config_path = Path("config/settings.yaml")
    config = load_config(config_path)
    
    # Initialize TTS service
    print("\n🔊 Initializing TTS service...")
    tts = EmotionalTTSService(config)
    tts.load_model()
    print("✅ TTS service ready!")
    
    # Demo questions
    questions = [
        "What is motor learning?",
        "How does gait adaptation work?",
        "Explain the principles of biomechanics.",
    ]
    
    print("\n📚 Demonstrating RAG + TTS Integration")
    print("-" * 40)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}]")
        print(f"👤 User: {question}")
        
        # Simulate RAG processing
        print("🔍 Retrieving relevant documents...")
        time.sleep(0.5)  # Simulate retrieval time
        
        print("🤔 Generating response...")
        response = simulate_rag_response(question)
        time.sleep(0.3)  # Simulate generation time
        
        print(f"🤖 Assistant: {response[:100]}...")
        
        # Generate TTS
        print("🎙️ Converting to speech...")
        
        # Determine emotion based on content
        if "remarkably" in response.lower() or "fascinating" in response.lower():
            emotion = "excited"
        elif "crucial" in response.lower() or "essential" in response.lower():
            emotion = "emphatic"
        else:
            emotion = "neutral"
        
        audio_path = tts.synthesize_with_emotion(
            response,
            emotion=emotion,
            output_path=Path(config["tts"]["output_dir"]) / f"rag_response_{i}.wav"
        )
        
        print(f"✅ Audio saved to: {audio_path}")
        
        # Play audio
        if input("   Play audio? (y/n): ").lower() == 'y':
            play_audio(audio_path)
        
        if i < len(questions):
            input("\nPress Enter for next question...")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("💬 Interactive Mode (type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        question = input("\n👤 Your question: ").strip()
        
        if question.lower() in ["exit", "quit"]:
            break
        
        if not question:
            continue
        
        # Process question
        print("🔍 Processing...")
        response = simulate_rag_response(question)
        
        print(f"\n🤖 {response}\n")
        
        # Generate TTS
        if input("Generate audio? (y/n): ").lower() == 'y':
            print("🎙️ Generating speech...")
            
            # Let user choose emotion
            print("Choose emotion: neutral, excited, dramatic, calm, emphatic")
            emotion = input("Emotion (default=neutral): ").strip() or "neutral"
            
            audio_path = tts.synthesize_with_emotion(response, emotion=emotion)
            print(f"✅ Audio saved to: {audio_path}")
            
            if input("Play audio? (y/n): ").lower() == 'y':
                play_audio(audio_path)
    
    # Cleanup
    print("\n👋 Cleaning up...")
    tts.cleanup()
    print("✨ Demo complete!")


if __name__ == "__main__":
    main()