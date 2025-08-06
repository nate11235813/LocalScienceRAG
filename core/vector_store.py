"""Vector store management for document retrieval."""

from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages FAISS vector store for document retrieval."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.index_dir = Path(config["vector_store"]["index_dir"])
        self.vector_store: Optional[FAISS] = None
    
    def create_vector_store(
        self, 
        documents: List[Document], 
        embeddings: HuggingFaceEmbeddings
    ) -> FAISS:
        """Create a new vector store from documents.
        
        Args:
            documents: List of documents to index
            embeddings: Embeddings model
            
        Returns:
            FAISS vector store
        """
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            self.vector_store = FAISS.from_documents(documents, embeddings)
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def save_vector_store(self, path: Optional[Path] = None) -> None:
        """Save vector store to disk.
        
        Args:
            path: Optional path to save to (defaults to config path)
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        save_path = path or self.index_dir
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.vector_store.save_local(str(save_path))
            size_mb = (save_path / "index.faiss").stat().st_size / 1_048_576
            logger.info(f"Saved vector store to {save_path} ({size_mb:.1f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load_vector_store(
        self, 
        embeddings: HuggingFaceEmbeddings,
        path: Optional[Path] = None
    ) -> FAISS:
        """Load vector store from disk.
        
        Args:
            embeddings: Embeddings model
            path: Optional path to load from (defaults to config path)
            
        Returns:
            FAISS vector store
        """
        load_path = path or self.index_dir
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        try:
            logger.info(f"Loading vector store from {load_path}")
            self.vector_store = FAISS.load_local(
                str(load_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded")
        
        k = k or self.config["vector_store"]["top_k"]
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def get_vector_store(self) -> Optional[FAISS]:
        """Get the loaded vector store.
        
        Returns:
            FAISS vector store or None
        """
        return self.vector_store
    
    def cleanup(self) -> None:
        """Clean up vector store resources."""
        if self.vector_store is not None:
            del self.vector_store
            self.vector_store = None
            logger.info("Vector store resources cleaned up")