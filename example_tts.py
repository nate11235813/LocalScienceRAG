#!/usr/bin/env python
"""Example script demonstrating Chatterbox TTS usage."""

import argparse
from pathlib import Path
from utils import load_config
from services import TTSService, EmotionalTTSService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_tts_example(config):
    """Basic TTS example with default settings."""
    print("🎯 Basic TTS Example")
    print("-" * 40)
    
    # Initialize TTS service
    tts = TTSService(config)
    tts.load_model()
    
    # Sample texts
    texts = [
        "Welcome to the Chatterbox text-to-speech demonstration.",
        "This system can convert any text into natural sounding speech.",
        "The model supports emotion control and voice cloning features."
    ]
    
    # Generate speech for each text
    for i, text in enumerate(texts, 1):
        print(f"\n📝 Text {i}: {text}")
        output_path = tts.synthesize(text)
        print(f"✅ Audio saved to: {output_path}")
    
    tts.cleanup()


def emotion_control_example(config):
    """Example demonstrating emotion control."""
    print("\n🎭 Emotion Control Example")
    print("-" * 40)
    
    # Initialize emotional TTS service
    tts = EmotionalTTSService(config)
    tts.load_model()
    
    text = "The experiments revealed fascinating insights about motor control and adaptation."
    
    # Generate with different emotions
    emotions = ["neutral", "excited", "dramatic", "calm", "emphatic"]
    
    for emotion in emotions:
        print(f"\n🎨 Emotion: {emotion}")
        output_path = Path(config["tts"]["output_dir"]) / f"emotion_{emotion}.wav"
        tts.synthesize_with_emotion(text, emotion=emotion, output_path=output_path)
        print(f"✅ Saved to: {output_path}")
    
    tts.cleanup()


def scientific_text_example(config):
    """Example with scientific text from biomechanics."""
    print("\n🔬 Scientific Text Example")
    print("-" * 40)
    
    tts = TTSService(config)
    tts.load_model()
    
    # Scientific texts with appropriate emotion settings
    examples = [
        {
            "text": "The center of mass trajectory during human locomotion follows a sinusoidal pattern.",
            "exaggeration": 0.4,  # Calm, professional
            "cfg_weight": 0.6
        },
        {
            "text": "Remarkably, the experiments showed a 40% improvement in adaptation rates!",
            "exaggeration": 0.7,  # More excited
            "cfg_weight": 0.3
        },
        {
            "text": "The cerebellum plays a crucial role in motor learning and coordination.",
            "exaggeration": 0.5,  # Neutral
            "cfg_weight": 0.5
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📖 Example {i}:")
        print(f"   Text: {example['text'][:60]}...")
        print(f"   Settings: exaggeration={example['exaggeration']}, cfg={example['cfg_weight']}")
        
        output_path = Path(config["tts"]["output_dir"]) / f"scientific_{i}.wav"
        tts.synthesize(
            example["text"],
            output_path=output_path,
            exaggeration=example["exaggeration"],
            cfg_weight=example["cfg_weight"]
        )
        print(f"   ✅ Saved to: {output_path}")
    
    tts.cleanup()


def voice_cloning_example(config, voice_sample_path):
    """Example demonstrating voice cloning."""
    print("\n🎤 Voice Cloning Example")
    print("-" * 40)
    
    if not Path(voice_sample_path).exists():
        print(f"❌ Voice sample not found: {voice_sample_path}")
        print("   Please provide a valid audio file for voice cloning.")
        return
    
    tts = TTSService(config)
    tts.load_model()
    
    text = "This speech is generated using your voice sample as a reference."
    
    print(f"📎 Using voice sample: {voice_sample_path}")
    print(f"📝 Text: {text}")
    
    output_path = tts.convert_voice(
        text,
        voice_sample_path,
        output_path=Path(config["tts"]["output_dir"]) / "voice_cloned.wav"
    )
    
    print(f"✅ Cloned voice audio saved to: {output_path}")
    
    tts.cleanup()


def batch_processing_example(config):
    """Example of batch text processing."""
    print("\n📚 Batch Processing Example")
    print("-" * 40)
    
    tts = TTSService(config)
    tts.load_model()
    
    # Multiple texts to process
    texts = [
        "Biomechanics is the study of mechanical laws relating to movement.",
        "Gait analysis reveals patterns in human locomotion.",
        "Sensory feedback is essential for motor control.",
        "The vestibular system helps maintain balance and posture.",
        "Muscle synergies simplify the control of complex movements."
    ]
    
    print(f"Processing {len(texts)} texts...")
    
    output_dir = Path(config["tts"]["output_dir"]) / "batch"
    paths = tts.synthesize_batch(texts, output_dir=output_dir)
    
    print(f"\n✅ Batch processing complete!")
    print(f"   Generated {len([p for p in paths if p])} audio files")
    print(f"   Output directory: {output_dir}")
    
    tts.cleanup()


def main():
    """Main entry point for TTS examples."""
    parser = argparse.ArgumentParser(
        description="Chatterbox TTS Examples for Scientific RAG"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--example",
        choices=["basic", "emotion", "scientific", "voice", "batch", "all"],
        default="basic",
        help="Which example to run"
    )
    
    parser.add_argument(
        "--voice-sample",
        type=Path,
        help="Path to voice sample for cloning (required for voice example)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Ensure output directory exists
    output_dir = Path(config["tts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎙️ Chatterbox TTS Examples")
    print("=" * 50)
    
    # Run selected examples
    if args.example == "basic" or args.example == "all":
        basic_tts_example(config)
    
    if args.example == "emotion" or args.example == "all":
        emotion_control_example(config)
    
    if args.example == "scientific" or args.example == "all":
        scientific_text_example(config)
    
    if args.example == "voice":
        if args.voice_sample:
            voice_cloning_example(config, args.voice_sample)
        else:
            print("\n❌ Voice cloning example requires --voice-sample argument")
    
    if args.example == "batch" or args.example == "all":
        batch_processing_example(config)
    
    print("\n✨ Examples complete!")


if __name__ == "__main__":
    main()