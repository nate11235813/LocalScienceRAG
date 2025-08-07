"""Model management for GPT-OSS-20B."""

import os
import torch
from typing import Optional, Dict, Any, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Pipeline,
    pipeline,
)
import logging
from .gpt_oss_loader import GPTOSSLoader

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the loading and inference of GPT-OSS-20B model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model manager with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_id = config["model"]["id"]
        self.device = config["model"]["device"]
        self.dtype = self._get_dtype(config["model"]["dtype"])
        
        # Set environment variables
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.pipeline: Optional[Pipeline] = None
        
    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            # Check if this is a GPT-OSS model
            is_gpt_oss = "gpt-oss" in self.model_id.lower()
            
            if is_gpt_oss:
                logger.info(f"Detected GPT-OSS model: {self.model_id}")
                logger.info("Using custom GPT-OSS loader...")
                
                # Use custom loader for GPT-OSS models
                self.model, self.tokenizer = GPTOSSLoader.load_model_and_tokenizer(
                    model_id=self.model_id,
                    device=self.device,
                    dtype=self.dtype,
                    max_seq_length=self.config["model"].get("max_seq_length", 2048),
                    load_in_4bit=self.config["model"].get("load_in_4bit", False),
                    load_in_8bit=self.config["model"].get("load_in_8bit", False),
                )
            else:
                # Standard loading for non-GPT-OSS models
                logger.info(f"Loading tokenizer from {self.model_id}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, 
                    trust_remote_code=True
                )
                
                logger.info(f"Loading model from {self.model_id}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            
            # Free CPU weight copy to reclaim memory
            if hasattr(self.model, "_cpu_state_dict"):
                delattr(self.model, "_cpu_state_dict")
                logger.debug("Freed CPU state dict to save memory")
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device,
                torch_dtype=self.dtype,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self, 
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """Generate text using the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config["model"]["max_new_tokens"]
        temperature = temperature or self.config["model"]["temperature"]
        do_sample = do_sample if do_sample is not None else self.config["model"]["do_sample"]
        
        try:
            output = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            return output[0]["generated_text"]
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def apply_chat_template(
        self, 
        messages: List[Dict[str, str]], 
        add_generation_prompt: bool = True
    ) -> str:
        """Apply chat template to messages.
        
        Args:
            messages: List of message dictionaries
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Formatted prompt
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
            
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        torch.cuda.empty_cache()
        logger.info("Model resources cleaned up")