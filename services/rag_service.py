"""RAG service for question answering."""

import re
from typing import Dict, Any, List, Optional, Tuple
from core import ModelManager, DocumentRetriever
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG-based question answering."""
    
    # Pattern to remove analysis sections from responses
    ANALYSIS_PATTERN = re.compile(
        r"(?i)^analysis.*?(?:assistantfinal|assistant:|\n\n)", 
        re.S
    )
    
    def __init__(
        self,
        model_manager: ModelManager,
        retriever: DocumentRetriever,
        config: Dict[str, Any]
    ):
        """Initialize RAG service.
        
        Args:
            model_manager: Model manager instance
            retriever: Document retriever instance
            config: Configuration dictionary
        """
        self.model_manager = model_manager
        self.retriever = retriever
        self.config = config
        self.system_prompt = config["ui"]["system_prompt"]
    
    def build_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        k: Optional[int] = None
    ) -> str:
        """Build prompt with retrieved context.
        
        Args:
            query: User query
            context: Optional pre-retrieved context
            k: Number of documents to retrieve (if context not provided)
            
        Returns:
            Formatted prompt
        """
        # Retrieve context if not provided
        if context is None:
            _, context = self.retriever.retrieve_context(query, k)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": context},
            {"role": "user", "content": query},
        ]
        
        # Apply chat template
        prompt = self.model_manager.apply_chat_template(messages)
        return prompt
    
    def clean_response(self, raw_response: str, prompt_length: int) -> str:
        """Clean the model response.
        
        Args:
            raw_response: Raw model output
            prompt_length: Length of the input prompt
            
        Returns:
            Cleaned response
        """
        # Extract only the generated part
        response = raw_response[prompt_length:]
        
        # Remove analysis sections
        response = re.sub(self.ANALYSIS_PATTERN, "", response)
        
        # Remove assistantfinal markers
        response = re.sub(r"(?i)assistantfinal[:\s]*", "", response, count=1)
        
        return response.strip()
    
    def answer_question(
        self,
        query: str,
        k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """Answer a question using RAG.
        
        Args:
            query: User question
            k: Number of documents to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_context: Whether to return retrieved context
            
        Returns:
            Dictionary with answer and optional metadata
        """
        try:
            # Retrieve context
            documents, context = self.retriever.retrieve_context(query, k)
            logger.info(f"Retrieved {len(documents)} documents for query")
            
            # Build prompt
            prompt = self.build_prompt(query, context=context)
            prompt_length = len(prompt)
            
            # Generate response
            logger.info("Generating response...")
            raw_response = self.model_manager.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # Clean response
            answer = self.clean_response(raw_response, prompt_length)
            
            # Build result
            result = {
                "answer": answer,
                "query": query,
            }
            
            if return_context:
                result["context"] = context
                result["documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
            
            logger.info("Successfully generated answer")
            return result
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise
    
    def answer_batch(
        self,
        queries: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions.
        
        Args:
            queries: List of questions
            **kwargs: Additional arguments for answer_question
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing question {i}/{len(queries)}")
            try:
                result = self.answer_question(query, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to answer question {i}: {e}")
                results.append({
                    "answer": f"Error: {str(e)}",
                    "query": query,
                    "error": True
                })
        
        return results