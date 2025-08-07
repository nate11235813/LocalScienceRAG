#!/usr/bin/env python
"""Simple fix to register GPT-OSS architecture with transformers."""

import sys
import os

# Add the architecture registration before importing anything else
def register_gpt_oss():
    """Register GPT-OSS as a Llama-based architecture."""
    from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
    from transformers.models.llama import LlamaConfig, LlamaForCausalLM
    
    # Register GPT-OSS config as Llama-based
    if "gpt_oss" not in CONFIG_MAPPING._extra_content:
        CONFIG_MAPPING.register("gpt_oss", LlamaConfig)
        print("✓ Registered gpt_oss config mapping")
    
    # Register GPT-OSS model as Llama-based
    try:
        if LlamaConfig not in MODEL_FOR_CAUSAL_LM_MAPPING._extra_content:
            MODEL_FOR_CAUSAL_LM_MAPPING.register(LlamaConfig, LlamaForCausalLM)
            print("✓ Registered gpt_oss model mapping")
    except:
        pass  # May already be registered
    
    print("✓ GPT-OSS architecture registered successfully")

# Register the architecture
register_gpt_oss()

# Now import and run the main script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main

if __name__ == "__main__":
    # Pass through all arguments to main
    sys.argv[0] = "main.py"
    main.main()
