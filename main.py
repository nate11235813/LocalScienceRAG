#!/usr/bin/env python
"""Main entry point for the RAG application."""

import argparse
import sys
from pathlib import Path
from utils import load_config, setup_logging
from core import ModelManager, EmbeddingsManager, VectorStoreManager, DocumentRetriever
from services import DocumentIndexer, RAGService
import logging

logger = logging.getLogger(__name__)


def build_index_command(args, config):
    """Handle the build-index command."""
    try:
        # Initialize components
        embeddings_manager = EmbeddingsManager(config)
        vector_store_manager = VectorStoreManager(config)
        
        # Create indexer
        indexer = DocumentIndexer(
            embeddings_manager,
            vector_store_manager,
            config
        )
        
        # Build index
        indexer.build_index()
        
        print("✅ Index built successfully!")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        sys.exit(1)


def chat_command(args, config):
    """Handle the chat command."""
    import threading
    import itertools
    import time
    
    def spinner(stop_event):
        """Display spinner while processing."""
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r🤖 thinking… {ch}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r")
    
    try:
        print("🔗 Loading components...")
        
        # Initialize components
        embeddings_manager = EmbeddingsManager(config)
        embeddings = embeddings_manager.load_embeddings()
        
        vector_store_manager = VectorStoreManager(config)
        vector_store_manager.load_vector_store(embeddings)
        
        retriever = DocumentRetriever(vector_store_manager, config)
        
        print("🧠 Loading model (this may take a minute)...")
        model_manager = ModelManager(config)
        model_manager.load_model()
        
        # Create RAG service
        rag_service = RAGService(model_manager, retriever, config)
        
        print("\n📝 Ask me anything about your PDFs (type 'exit' to quit)\n")
        
        # Interactive loop
        while True:
            try:
                question = input("👤> ").strip()
                
                if question.lower() in {"exit", "quit"}:
                    break
                
                if not question:
                    continue
                
                # Start spinner if enabled
                if config["ui"]["show_spinner"]:
                    stop_spin = threading.Event()
                    thr = threading.Thread(target=spinner, args=(stop_spin,), daemon=True)
                    thr.start()
                
                # Get answer
                result = rag_service.answer_question(
                    question,
                    return_context=args.show_context
                )
                
                # Stop spinner
                if config["ui"]["show_spinner"]:
                    stop_spin.set()
                    thr.join()
                
                # Display answer
                print(f"🤖 {result['answer']}\n")
                
                # Optionally show context
                if args.show_context:
                    print("📚 Retrieved context:")
                    for i, doc in enumerate(result["documents"], 1):
                        print(f"  [{i}] {doc['content'][:200]}...")
                    print()
                
            except KeyboardInterrupt:
                break
        
        # Cleanup
        print("\n👋 Goodbye!")
        model_manager.cleanup()
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        sys.exit(1)


def test_command(args, config):
    """Handle the test command."""
    try:
        print("🧠 Loading model...")
        model_manager = ModelManager(config)
        model_manager.load_model()
        
        # Build test prompt
        messages = [
            {"role": "user", "content": args.prompt or "Explain extrapolated center of mass and how it is used in biomechanics."}
        ]
        
        prompt = model_manager.apply_chat_template(messages)
        
        print("🤖 Generating response...")
        response = model_manager.generate(
            prompt,
            max_new_tokens=args.max_tokens or config["model"]["max_new_tokens"]
        )
        
        # Extract completion
        generated = response.split(prompt, 1)[-1].strip()
        print(f"\n{generated}\n")
        
        # Cleanup
        model_manager.cleanup()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Application for Scientific PDFs"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build index command
    build_parser = subparsers.add_parser("build-index", help="Build vector index from PDFs")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with RAG")
    chat_parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context with answers"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test model generation")
    test_parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt to test"
    )
    test_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config, args.log_file, args.verbose)
    
    # Execute command
    if args.command == "build-index":
        build_index_command(args, config)
    elif args.command == "chat":
        chat_command(args, config)
    elif args.command == "test":
        test_command(args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()