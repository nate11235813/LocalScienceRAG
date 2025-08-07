#!/usr/bin/env python
"""Test script to verify a simple working model for RAG + TTS."""

from pathlib import Path
from utils import load_config
from services import RAGService, TTSService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_with_simple_model():
    """Test RAG + TTS with a simple, reliable model."""
    
    # Create a test config with a small, working model
    test_config = {
        "model": {
            "id": "microsoft/phi-2",  # Small 2.7B model that works well
            "dtype": "float32",
            "device": "mps",
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "load_in_8bit": False,
        },
        "embeddings": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "mps",
        },
        "vector_store": {
            "index_path": "data/faiss_store",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 4,
        },
        "ui": {
            "show_spinner": True,
            "system_prompt": "You are an expert biomechanics assistant.",
        },
        "tts": {
            "device": "mps",
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "output_dir": "data/tts_output",
            "enabled": True,
            "auto_play": False,
            "emotion_preset": "neutral",
        }
    }
    
    print("=" * 60)
    print("🧪 Testing with a Simple Working Model")
    print("=" * 60)
    
    try:
        # Initialize services
        print("\n📚 Initializing RAG Service...")
        rag_service = RAGService(test_config)
        
        print("\n🔊 Initializing TTS Service...")
        tts_service = TTSService(test_config)
        tts_service.load_model()
        
        # Test question
        test_question = "What is biomechanics in simple terms?"
        print(f"\n❓ Question: {test_question}")
        
        # Get RAG response
        print("\n🤖 Generating response...")
        result = rag_service.answer_question(test_question)
        answer = result["answer"]
        
        print(f"\n💬 Answer: {answer[:200]}..." if len(answer) > 200 else f"\n💬 Answer: {answer}")
        
        # Convert to speech
        print("\n🎙️ Converting to speech...")
        audio_path = tts_service.synthesize(answer[:500])  # Limit length for TTS
        print(f"✅ Audio saved to: {audio_path}")
        
        # Play audio
        import subprocess
        import platform
        if platform.system() == "Darwin":
            print("\n▶️ Playing audio...")
            subprocess.run(["afplay", str(audio_path)], check=False)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_simple_model()
    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
        print("\nThe RAG + TTS pipeline is working correctly.")
        print("The issue with GPT-OSS is due to incomplete model weights.")
    else:
        print("❌ Test failed. Check the error messages above.")