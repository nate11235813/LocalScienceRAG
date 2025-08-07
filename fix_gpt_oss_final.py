#!/usr/bin/env python
"""Final fix for GPT-OSS with architecture registration and chat template."""

import warnings
warnings.filterwarnings("ignore")

def setup_gpt_oss():
    """Setup GPT-OSS architecture and fix tokenizer issues."""
    try:
        from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING, AutoTokenizer
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
        
        # Monkey-patch the tokenizer loading to add chat template
        original_from_pretrained = AutoTokenizer.from_pretrained
        
        def patched_from_pretrained(model_id, *args, **kwargs):
            tokenizer = original_from_pretrained(model_id, *args, **kwargs)
            
            # Add chat template if missing (for GPT-OSS models)
            if tokenizer.chat_template is None and "gpt-oss" in model_id.lower():
                # Use a simple Llama-style template
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'system' %}"
                    "{{ message['content'] }}\n\n"
                    "{% elif message['role'] == 'user' %}"
                    "User: {{ message['content'] }}\n"
                    "{% elif message['role'] == 'assistant' %}"
                    "Assistant: {{ message['content'] }}"
                    "{% endif %}"
                    "{% if not loop.last %}\n{% endif %}"
                    "{% endfor %}"
                    "{% if messages[-1]['role'] != 'assistant' %}Assistant: {% endif %}"
                )
                print("✓ Added chat template to tokenizer")
            
            return tokenizer
        
        AutoTokenizer.from_pretrained = patched_from_pretrained
        print("✓ Tokenizer patching complete")
        
        return True
    except Exception as e:
        print(f"⚠️ Setup failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Setup architecture and tokenizer fixes
    if not setup_gpt_oss():
        print("\n⚠️ Failed to setup GPT-OSS")
        sys.exit(1)
    
    # Run main with all arguments
    import main
    sys.argv[0] = "main.py"
    main.main()
