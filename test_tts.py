#!/usr/bin/env python
"""Simple test script for Chatterbox TTS integration."""

import sys
from pathlib import Path


def test_import():
    """Test if Chatterbox TTS can be imported."""
    print("Testing Chatterbox TTS import...")
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ Chatterbox TTS imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Chatterbox TTS: {e}")
        print("\nPlease install with: pip install chatterbox-tts")
        return False


def test_basic_synthesis():
    """Test basic TTS synthesis."""
    print("\nTesting basic synthesis...")
    
    try:
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS
        
        # Check device availability
        if torch.cuda.is_available():
            device = "cuda"
            print(f"  Using CUDA device")
        elif torch.backends.mps.is_available():
            device = "mps"
            print(f"  Using MPS device (Apple Silicon)")
        else:
            device = "cpu"
            print(f"  Using CPU device")
        
        # Load model
        print(f"  Loading model on {device}...")
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Generate speech
        text = "Hello! This is a test of the Chatterbox text-to-speech system."
        print(f"  Generating speech for: '{text}'")
        
        wav = model.generate(text)
        
        # Save audio
        output_path = "test_output.wav"
        torchaudio.save(output_path, wav, model.sr)
        
        print(f"✅ Audio generated and saved to: {output_path}")
        print(f"  Sample rate: {model.sr} Hz")
        print(f"  Audio shape: {wav.shape}")
        
        # Clean up
        Path(output_path).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Synthesis test failed: {e}")
        return False


def test_service_integration():
    """Test TTS service integration."""
    print("\nTesting TTS service integration...")
    
    try:
        from utils import load_config
        from services import TTSService
        
        # Load config
        config_path = Path("config/settings.yaml")
        if not config_path.exists():
            print(f"  Warning: Config file not found at {config_path}")
            # Use minimal config
            config = {
                "tts": {
                    "device": "cpu",
                    "exaggeration": 0.5,
                    "cfg_weight": 0.5,
                    "output_dir": "data/tts_output"
                }
            }
        else:
            config = load_config(config_path)
        
        # Initialize service
        print("  Initializing TTS service...")
        tts = TTSService(config)
        tts.load_model()
        
        # Test synthesis
        text = "The TTS service is working correctly."
        print(f"  Synthesizing: '{text}'")
        
        output_path = tts.synthesize(text)
        
        print(f"✅ TTS service test successful")
        print(f"  Output saved to: {output_path}")
        
        # Clean up
        tts.cleanup()
        if output_path and Path(output_path).exists():
            Path(output_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Service integration test failed: {e}")
        return False


def test_emotion_presets():
    """Test emotion presets."""
    print("\nTesting emotion presets...")
    
    try:
        from services import EmotionalTTSService
        
        # Check available presets
        presets = EmotionalTTSService.EMOTION_PRESETS
        print(f"  Available emotion presets: {list(presets.keys())}")
        
        for emotion, settings in presets.items():
            print(f"    {emotion}: exaggeration={settings['exaggeration']}, cfg={settings['cfg_weight']}")
        
        print("✅ Emotion presets validated")
        return True
        
    except Exception as e:
        print(f"❌ Emotion preset test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("🧪 Chatterbox TTS Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Basic Synthesis", test_basic_synthesis),
        ("Service Integration", test_service_integration),
        ("Emotion Presets", test_emotion_presets),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n[{name}]")
        print("-" * 30)
        success = test_func()
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())