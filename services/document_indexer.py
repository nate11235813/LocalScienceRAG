"""Document indexing service for building vector stores."""

from pathlib import Path
from typing import Dict, Any, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from core import EmbeddingsManager, VectorStoreManager
import logging

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Service for indexing documents into vector store."""
    
    def __init__(
        self, 
        embeddings_manager: EmbeddingsManager,
        vector_store_manager: VectorStoreManager,
        config: Dict[str, Any]
    ):
        """Initialize document indexer.
        
        Args:
            embeddings_manager: Embeddings manager instance
            vector_store_manager: Vector store manager instance
            config: Configuration dictionary
        """
        self.embeddings_manager = embeddings_manager
        self.vector_store_manager = vector_store_manager
        self.config = config
        
        self.pdf_dir = Path(config["paths"]["pdf_dir"])
        self.chunk_size = config["vector_store"]["chunk_size"]
        self.chunk_overlap = config["vector_store"]["chunk_overlap"]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_and_split_pdfs(self) -> List[Document]:
        """Load and split PDF documents.
        
        Returns:
            List of document chunks
        """
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")
        
        documents = []
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_dir}")
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        for pdf_path in sorted(pdf_files):
            try:
                # Load PDF
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(pages)
                
                # Add source metadata
                for chunk in chunks:
                    chunk.metadata["source_file"] = pdf_path.name
                
                documents.extend(chunks)
                logger.info(f"  • {pdf_path.name:40s} → {len(chunks):4d} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(documents):,}")
        return documents
    
    def build_index(self) -> None:
        """Build the complete vector index from PDFs."""
        try:
            # Load and split documents
            logger.info("Loading and splitting PDFs...")
            documents = self.load_and_split_pdfs()
            
            if not documents:
                raise ValueError("No documents to index")
            
            # Load embeddings
            logger.info("Loading embeddings model...")
            embeddings = self.embeddings_manager.load_embeddings()
            
            # Create vector store
            logger.info("Creating vector store...")
            self.vector_store_manager.create_vector_store(documents, embeddings)
            
            # Save to disk
            logger.info("Saving vector store...")
            self.vector_store_manager.save_vector_store()
            
            logger.info("Index building completed successfully")
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            raise
    
    def update_index(self, new_pdf_paths: List[Path]) -> None:
        """Update existing index with new PDFs.
        
        Args:
            new_pdf_paths: List of paths to new PDF files
        """
        try:
            # Load existing vector store
            embeddings = self.embeddings_manager.get_embeddings()
            self.vector_store_manager.load_vector_store(embeddings)
            
            # Process new PDFs
            new_documents = []
            for pdf_path in new_pdf_paths:
                if not pdf_path.exists():
                    logger.warning(f"PDF not found: {pdf_path}")
                    continue
                
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                chunks = self.text_splitter.split_documents(pages)
                
                for chunk in chunks:
                    chunk.metadata["source_file"] = pdf_path.name
                
                new_documents.extend(chunks)
                logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks")
            
            if new_documents:
                # Add to existing vector store
                vector_store = self.vector_store_manager.get_vector_store()
                vector_store.add_documents(new_documents)
                
                # Save updated store
                self.vector_store_manager.save_vector_store()
                logger.info(f"Added {len(new_documents)} new chunks to index")
            else:
                logger.warning("No new documents to add")
                
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            raise