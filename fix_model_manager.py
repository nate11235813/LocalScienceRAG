#!/usr/bin/env python
"""Fix model_manager.py to use standard loading for all models."""

with open("core/model_manager.py", "r") as f:
    content = f.read()

# Find and replace the load_model method
old_method = '''    def load_model(self) -> None:
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
                )'''

new_method = '''    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
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
            )'''

content = content.replace(old_method, new_method)

with open("core/model_manager.py", "w") as f:
    f.write(content)

print("✅ Fixed model_manager.py to use standard loading for all models")
