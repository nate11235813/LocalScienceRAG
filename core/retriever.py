"""Document retrieval for RAG."""

from typing import Dict, Any, List, Tuple
from langchain_core.documents import Document
from .vector_store import VectorStoreManager
import logging

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Handles document retrieval for RAG."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, config: Dict[str, Any]):
        """Initialize document retriever.
        
        Args:
            vector_store_manager: Vector store manager instance
            config: Configuration dictionary
        """
        self.vector_store_manager = vector_store_manager
        self.config = config
        self.default_k = config["vector_store"]["top_k"]
    
    def retrieve_context(
        self, 
        query: str, 
        k: int = None
    ) -> Tuple[List[Document], str]:
        """Retrieve relevant context for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, formatted context string)
        """
        k = k or self.default_k
        
        try:
            # Retrieve similar documents
            documents = self.vector_store_manager.similarity_search(query, k=k)
            
            # Format context with citations
            context_parts = []
            for i, doc in enumerate(documents):
                citation = f"[{i+1}]"
                content = doc.page_content.strip()
                context_parts.append(f"{citation} {content}")
            
            formatted_context = "\n\n".join(context_parts)
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents, formatted_context
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise
    
    def format_with_metadata(
        self, 
        documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Format documents with their metadata.
        
        Args:
            documents: List of documents
            
        Returns:
            List of formatted document dictionaries
        """
        formatted = []
        for i, doc in enumerate(documents):
            formatted.append({
                "citation": f"[{i+1}]",
                "content": doc.page_content.strip(),
                "metadata": doc.metadata,
            })
        return formatted