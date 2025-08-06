"""Embeddings management for document vectorization."""

import os
from typing import Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages embeddings model for document vectorization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize embeddings manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config["embeddings"]["model_name"]
        self.device = config["embeddings"]["device"]
        self.embeddings_model: Optional[HuggingFaceEmbeddings] = None
        
        # Set environment variables
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    def load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load and return the embeddings model.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        if self.embeddings_model is None:
            try:
                logger.info(f"Loading embeddings model: {self.model_name}")
                self.embeddings_model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device},
                )
                logger.info("Embeddings model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embeddings model: {e}")
                raise
        
        return self.embeddings_model
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get the loaded embeddings model.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        if self.embeddings_model is None:
            return self.load_embeddings()
        return self.embeddings_model
    
    def cleanup(self) -> None:
        """Clean up embeddings resources."""
        if self.embeddings_model is not None:
            del self.embeddings_model
            self.embeddings_model = None
            logger.info("Embeddings resources cleaned up")