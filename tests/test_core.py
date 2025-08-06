"""Unit tests for core modules."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import yaml
from core import ModelManager, EmbeddingsManager, VectorStoreManager, DocumentRetriever


class TestModelManager(unittest.TestCase):
    """Tests for ModelManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model": {
                "id": "test-model",
                "dtype": "float32",
                "device": "cpu",
                "max_new_tokens": 100,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        self.model_manager = ModelManager(self.config)
    
    def test_initialization(self):
        """Test ModelManager initialization."""
        self.assertEqual(self.model_manager.model_id, "test-model")
        self.assertEqual(self.model_manager.device, "cpu")
        self.assertIsNone(self.model_manager.model)
        self.assertIsNone(self.model_manager.tokenizer)
    
    @patch('core.model_manager.AutoTokenizer')
    @patch('core.model_manager.AutoModelForCausalLM')
    def test_load_model(self, mock_model_class, mock_tokenizer_class):
        """Test model loading."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        self.model_manager.load_model()
        
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
        self.assertIsNotNone(self.model_manager.tokenizer)
        self.assertIsNotNone(self.model_manager.model)
    
    def test_generate_without_model(self):
        """Test generation fails without loaded model."""
        with self.assertRaises(ValueError):
            self.model_manager.generate("test prompt")


class TestEmbeddingsManager(unittest.TestCase):
    """Tests for EmbeddingsManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "embeddings": {
                "model_name": "test-embeddings",
                "device": "cpu"
            }
        }
        self.embeddings_manager = EmbeddingsManager(self.config)
    
    def test_initialization(self):
        """Test EmbeddingsManager initialization."""
        self.assertEqual(self.embeddings_manager.model_name, "test-embeddings")
        self.assertEqual(self.embeddings_manager.device, "cpu")
        self.assertIsNone(self.embeddings_manager.embeddings_model)
    
    @patch('core.embeddings.HuggingFaceEmbeddings')
    def test_load_embeddings(self, mock_embeddings_class):
        """Test embeddings loading."""
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        result = self.embeddings_manager.load_embeddings()
        
        mock_embeddings_class.assert_called_once_with(
            model_name="test-embeddings",
            model_kwargs={"device": "cpu"}
        )
        self.assertEqual(result, mock_embeddings)
        self.assertEqual(self.embeddings_manager.embeddings_model, mock_embeddings)


class TestVectorStoreManager(unittest.TestCase):
    """Tests for VectorStoreManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "vector_store": {
                "index_dir": "test_index",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k": 3
            }
        }
        self.vector_store_manager = VectorStoreManager(self.config)
    
    def test_initialization(self):
        """Test VectorStoreManager initialization."""
        self.assertEqual(self.vector_store_manager.index_dir, Path("test_index"))
        self.assertIsNone(self.vector_store_manager.vector_store)
    
    @patch('core.vector_store.FAISS')
    def test_create_vector_store(self, mock_faiss):
        """Test vector store creation."""
        mock_documents = [Mock(), Mock()]
        mock_embeddings = Mock()
        mock_store = Mock()
        mock_faiss.from_documents.return_value = mock_store
        
        result = self.vector_store_manager.create_vector_store(
            mock_documents, 
            mock_embeddings
        )
        
        mock_faiss.from_documents.assert_called_once_with(
            mock_documents, 
            mock_embeddings
        )
        self.assertEqual(result, mock_store)
        self.assertEqual(self.vector_store_manager.vector_store, mock_store)
    
    def test_similarity_search_without_store(self):
        """Test similarity search fails without loaded store."""
        with self.assertRaises(ValueError):
            self.vector_store_manager.similarity_search("test query")


class TestDocumentRetriever(unittest.TestCase):
    """Tests for DocumentRetriever."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "vector_store": {
                "top_k": 3
            }
        }
        self.mock_vector_store_manager = Mock()
        self.retriever = DocumentRetriever(
            self.mock_vector_store_manager, 
            self.config
        )
    
    def test_initialization(self):
        """Test DocumentRetriever initialization."""
        self.assertEqual(self.retriever.default_k, 3)
        self.assertEqual(self.retriever.vector_store_manager, self.mock_vector_store_manager)
    
    def test_retrieve_context(self):
        """Test context retrieval."""
        # Mock documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "Content 1"
        mock_doc2 = Mock()
        mock_doc2.page_content = "Content 2"
        
        self.mock_vector_store_manager.similarity_search.return_value = [
            mock_doc1, mock_doc2
        ]
        
        docs, context = self.retriever.retrieve_context("test query", k=2)
        
        self.mock_vector_store_manager.similarity_search.assert_called_once_with(
            "test query", k=2
        )
        self.assertEqual(len(docs), 2)
        self.assertIn("[1] Content 1", context)
        self.assertIn("[2] Content 2", context)


if __name__ == "__main__":
    unittest.main()