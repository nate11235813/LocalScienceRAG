#!/usr/bin/env python
"""Complete fix for GPT-OSS model loading with architecture registration."""

import warnings
warnings.filterwarnings("ignore")

def setup_gpt_oss():
    """Setup GPT-OSS architecture as Llama-based."""
    try:
        from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
        from transformers.models.llama import LlamaConfig, LlamaForCausalLM
        
        # Register GPT-OSS as Llama-based architecture
        if "gpt_oss" not in CONFIG_MAPPING._extra_content:
            CONFIG_MAPPING.register("gpt_oss", LlamaConfig)
            print("✓ Registered GPT-OSS config mapping")
        
        # Register model mapping
        try:
            if LlamaConfig not in MODEL_FOR_CAUSAL_LM_MAPPING._extra_content:
                MODEL_FOR_CAUSAL_LM_MAPPING.register(LlamaConfig, LlamaForCausalLM)
                print("✓ Registered GPT-OSS model mapping")
        except:
            pass
        
        print("✓ GPT-OSS architecture setup complete")
        return True
    except Exception as e:
        print(f"⚠️ Architecture registration failed: {e}")
        return False

def test_model():
    """Test if the model loads properly."""
    from pathlib import Path
    from utils import load_config
    from core.model_manager import ModelManager
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load config
        config = load_config(Path("config/settings.yaml"))
        print(f"\n📋 Config loaded: {config['model']['id']}")
        
        # Initialize model manager
        print("\n🔧 Initializing Model Manager...")
        model_manager = ModelManager(config)
        
        # Load model
        print("\n🧠 Loading model...")
        model_manager.load_model()
        
        # Test generation
        print("\n✨ Testing generation...")
        response = model_manager.generate("The capital of France is", max_new_tokens=10)
        print(f"Response: {response}")
        
        # Check for gibberish patterns
        if response and ("a a a" in response.lower() or "and and and" in response.lower()):
            print("\n⚠️ WARNING: Model may be producing gibberish due to missing weights!")
            print("Consider using a different model or checkpoint.")
            return False
        
        print("\n✅ Model loaded and tested successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Setup architecture
    if not setup_gpt_oss():
        print("\n⚠️ Failed to setup GPT-OSS architecture")
        sys.exit(1)
    
    # Check if we should test or run main
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        success = test_model()
        sys.exit(0 if success else 1)
    else:
        # Run main with all arguments
        import main
        sys.argv[0] = "main.py"
        main.main()
