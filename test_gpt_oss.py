#!/usr/bin/env python
"""Test script for GPT-OSS model loading."""

import sys
from pathlib import Path
from utils import load_config
from core.model_manager import ModelManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gpt_oss_loading():
    """Test if GPT-OSS model can be loaded properly."""
    print("=" * 60)
    print("🧪 Testing GPT-OSS Model Loading")
    print("=" * 60)
    
    try:
        # Load configuration
        print("\n📁 Loading configuration...")
        config = load_config(Path("config/settings.yaml"))
        print(f"✅ Config loaded: {config['model']['id']}")
        
        # Initialize model manager
        print("\n🤖 Initializing Model Manager...")
        model_manager = ModelManager(config)
        print("✅ Model Manager initialized")
        
        # Try to load the model
        print("\n📦 Loading GPT-OSS model...")
        print(f"   Model ID: {config['model']['id']}")
        print(f"   Device: {config['model']['device']}")
        print(f"   Dtype: {config['model']['dtype']}")
        
        model_manager.load_model()
        
        print("\n✅ Model loaded successfully!")
        
        # Test basic generation
        print("\n🔮 Testing basic generation...")
        test_prompt = "Hello, this is a test. The capital of France is"
        
        try:
            response = model_manager.generate(
                test_prompt,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True
            )
            print(f"\n📝 Prompt: {test_prompt}")
            print(f"💬 Response: {response}")
            print("\n✅ Generation successful!")
            
        except Exception as e:
            print(f"\n⚠️ Generation failed: {e}")
            print("This might be due to memory constraints or model issues.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading GPT-OSS model: {e}")
        print("\n📋 Troubleshooting tips:")
        print("1. Check if the model ID is correct")
        print("2. Ensure you have enough memory/disk space")
        print("3. Check your internet connection for model download")
        print("4. Try setting device to 'cpu' in config if MPS fails")
        print("5. Consider using quantization (load_in_4bit or load_in_8bit)")
        
        import traceback
        print("\n🔍 Full error trace:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    success = test_gpt_oss_loading()
    sys.exit(0 if success else 1)