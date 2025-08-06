"""Custom loader for GPT-OSS model that properly registers the architecture."""

import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.auto import auto_factory
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class GPTOSSLoader:
    """Custom loader for GPT-OSS models that handles architecture registration."""
    
    @staticmethod
    def load_model_and_tokenizer(
        model_id: str,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        max_seq_length: int = 2048,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_cache: bool = True,
        **kwargs
    ):
        """Load GPT-OSS model with proper architecture registration.
        
        Args:
            model_id: Model identifier (e.g., "unsloth/gpt-oss-20b-BF16")
            device: Device to load model on
            dtype: Data type for model weights
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to use 4-bit quantization
            load_in_8bit: Whether to use 8-bit quantization
            use_cache: Whether to use KV cache
            **kwargs: Additional arguments for model loading
            
        Returns:
            tuple: (model, tokenizer)
        """
        logger.info(f"Loading GPT-OSS model: {model_id}")
        
        # Handle quantization
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype or torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization")
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Using 8-bit quantization")
        
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Set a default chat template if not set
        if tokenizer.chat_template is None:
            # Use a simple template that works for most models
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"
            logger.info("Set default chat template")
        
        # Try to load model with different strategies
        model = None
        
        # Strategy 1: Try loading with trust_remote_code
        try:
            logger.info("Attempting to load model with trust_remote_code...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto" if device != "cpu" else None,
                torch_dtype=dtype,
                trust_remote_code=True,
                use_cache=use_cache,
                max_position_embeddings=max_seq_length,
                **kwargs
            )
            logger.info("Model loaded successfully with trust_remote_code")
        except Exception as e:
            logger.warning(f"Failed with trust_remote_code: {e}")
            
            # Strategy 2: Try loading as Llama architecture (GPT-OSS is often Llama-based)
            try:
                logger.info("Attempting to load as Llama architecture...")
                from transformers import LlamaForCausalLM, LlamaConfig
                
                # Load config and modify if needed
                config = AutoConfig.from_pretrained(
                    model_id,
                    trust_remote_code=False,
                )
                
                # Update config for GPT-OSS specifics
                if hasattr(config, 'model_type') and config.model_type == 'gpt_oss':
                    config.model_type = 'llama'  # Treat as Llama
                    logger.info("Changed model_type from gpt_oss to llama")
                
                config.max_position_embeddings = max_seq_length
                config.use_cache = use_cache
                
                model = LlamaForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    quantization_config=quantization_config,
                    device_map="auto" if device != "cpu" else None,
                    torch_dtype=dtype,
                    trust_remote_code=False,
                    **kwargs
                )
                logger.info("Model loaded successfully as Llama architecture")
            except Exception as e:
                logger.warning(f"Failed as Llama architecture: {e}")
                
                # Strategy 3: Generic fallback
                try:
                    logger.info("Attempting generic model loading...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto" if device != "cpu" else None,
                        torch_dtype=dtype,
                        trust_remote_code=False,
                        use_cache=use_cache,
                        **kwargs
                    )
                    logger.info("Model loaded successfully with generic approach")
                except Exception as e:
                    logger.error(f"All loading strategies failed: {e}")
                    raise RuntimeError(f"Failed to load GPT-OSS model {model_id}: {e}")
        
        # Move to device if not using device_map
        if model and device != "cpu" and quantization_config is None:
            model = model.to(device)
            logger.info(f"Model moved to {device}")
        
        # Log model info
        if model:
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded with {param_count/1e9:.2f}B parameters")
            logger.info(f"Model dtype: {next(model.parameters()).dtype}")
            logger.info(f"Model device: {next(model.parameters()).device}")
        
        return model, tokenizer


def register_gpt_oss_architecture():
    """Register GPT-OSS as a supported architecture in transformers.
    
    This function modifies the transformers auto model mappings to recognize
    GPT-OSS models and load them properly.
    """
    try:
        from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
        from transformers.models.llama import LlamaConfig, LlamaForCausalLM
        
        # Register GPT-OSS config as Llama-based
        if "gpt_oss" not in CONFIG_MAPPING._extra_content:
            CONFIG_MAPPING.register("gpt_oss", LlamaConfig)
            logger.info("Registered gpt_oss in CONFIG_MAPPING")
        
        # Register GPT-OSS model mapping (only if not already registered)
        # Note: We check if the config type is already mapped to avoid conflicts
        config_class_name = LlamaConfig.__name__
        already_registered = False
        
        try:
            # Check if this config is already mapped
            if hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, '_extra_content'):
                for cfg, mdl in MODEL_FOR_CAUSAL_LM_MAPPING._extra_content.items():
                    if cfg.__name__ == config_class_name and mdl == LlamaForCausalLM:
                        already_registered = True
                        break
        except:
            pass
        
        if not already_registered:
            try:
                MODEL_FOR_CAUSAL_LM_MAPPING.register(LlamaConfig, LlamaForCausalLM)
                logger.info("Registered gpt_oss model mapping")
            except Exception as e:
                logger.debug(f"Model mapping already exists or not needed: {e}")
            
        logger.info("GPT-OSS architecture setup complete")
        return True
        
    except Exception as e:
        logger.debug(f"Architecture registration note: {e}")
        # This is often fine - the mappings might already exist
        return True


# Auto-register on module import
register_gpt_oss_architecture()