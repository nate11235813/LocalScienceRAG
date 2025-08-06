#!/usr/bin/env python
"""Simple standalone Chatterbox TTS demo."""

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
import time

def main():
    print("=" * 50)
    print("🎙️ Chatterbox TTS Simple Demo")
    print("=" * 50)
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"✅ Using Apple Silicon (MPS)")
    else:
        device = "cpu"
        print(f"✅ Using CPU")
    
    # Load model
    print(f"\n📦 Loading Chatterbox TTS model on {device}...")
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"✅ Model loaded successfully (sample rate: {model.sr} Hz)")
    
    # Create output directory
    output_dir = Path("data/tts_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Example texts
    texts = [
        {
            "text": "Welcome to the LocalScienceRAG system with integrated text-to-speech capabilities.",
            "filename": "welcome.wav",
            "exaggeration": 0.5,
            "cfg_weight": 0.5
        },
        {
            "text": "The experiments revealed fascinating insights about motor control and adaptation in human biomechanics.",
            "filename": "scientific.wav",
            "exaggeration": 0.4,
            "cfg_weight": 0.6
        },
        {
            "text": "Remarkably, the cerebellum plays a crucial role in motor learning and coordination!",
            "filename": "excited.wav", 
            "exaggeration": 0.7,
            "cfg_weight": 0.3
        },
        {
            "text": "Gait analysis reveals complex patterns in human locomotion that adapt to various environmental conditions.",
            "filename": "calm.wav",
            "exaggeration": 0.3,
            "cfg_weight": 0.7
        }
    ]
    
    print(f"\n🔄 Generating {len(texts)} audio samples...")
    print("-" * 40)
    
    for i, item in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] Processing:")
        print(f"  📝 Text: '{item['text'][:60]}...'")
        print(f"  ⚙️ Settings: exaggeration={item['exaggeration']}, cfg={item['cfg_weight']}")
        
        # Generate audio
        start_time = time.time()
        wav = model.generate(
            item["text"],
            exaggeration=item["exaggeration"],
            cfg_weight=item["cfg_weight"]
        )
        generation_time = time.time() - start_time
        
        # Save audio
        output_path = output_dir / item["filename"]
        torchaudio.save(str(output_path), wav, model.sr)
        
        # Calculate duration
        duration = wav.shape[1] / model.sr
        
        print(f"  ✅ Generated in {generation_time:.2f}s")
        print(f"  📊 Audio duration: {duration:.2f}s")
        print(f"  💾 Saved to: {output_path}")
    
    print("\n" + "=" * 50)
    print("✨ Demo complete!")
    print(f"📁 All audio files saved to: {output_dir}")
    print("\n💡 Tips for using Chatterbox TTS:")
    print("  - Lower cfg_weight (0.3) + higher exaggeration (0.7) = more expressive")
    print("  - Higher cfg_weight (0.7) + lower exaggeration (0.3) = calmer speech")
    print("  - Default (0.5, 0.5) works well for most cases")
    
    # Cleanup
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

if __name__ == "__main__":
    main()