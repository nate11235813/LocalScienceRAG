# RAG Application - Refactored Architecture

## 📚 Overview

This is a RAG (Retrieval-Augmented Generation) application for scientific PDF processing. The architecture provides separation of concerns, modularity, testability, and maintainability.

## 🏗️ Architecture

### Directory Structure

```
├── config/
│   └── settings.yaml         # Configuration file
├── core/                     # Core business logic
│   ├── __init__.py
│   ├── model_manager.py      # LLM model management
│   ├── embeddings.py         # Embeddings management
│   ├── vector_store.py       # Vector store management
│   └── retriever.py          # Document retrieval
├── services/                 # Service layer
│   ├── __init__.py
│   ├── document_indexer.py   # PDF indexing service
│   ├── rag_service.py        # RAG question-answering service
│   └── tts_service.py        # Text-to-Speech service (Chatterbox TTS)
├── utils/                    # Utilities
│   ├── __init__.py
│   ├── config_loader.py      # Configuration loading
│   └── logging_config.py     # Logging setup
├── tests/                    # Unit tests
│   └── test_core.py          # Core module tests
├── data/
│   ├── science_pdfs/         # Input PDFs
│   └── faiss_store/          # Vector index storage
└── main.py                   # Main entry point

```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Additional refactoring dependencies
pip install pyyaml
```

### Configuration

Edit `config/settings.yaml` to customize:
- Model settings (ID, device, generation parameters)
- Embeddings configuration
- Vector store settings
- File paths
- Logging preferences

### Usage

```bash
# Build vector index from PDFs
python main.py build-index

# Interactive chat
python main.py chat

# Chat with context display
python main.py chat --show-context

# Chat with text-to-speech output
python main.py chat --tts

# Chat with TTS and auto-play audio
python main.py chat --tts --auto-play

# Test model generation
python main.py test --prompt "Your question here"

# Enable verbose logging
python main.py --verbose chat

# Save logs to file
python main.py --log-file logs/app.log chat
```

### Text-to-Speech (TTS) Features

The application now includes **Chatterbox TTS** integration for converting RAG responses to speech:

```bash
# Run TTS examples
python example_tts.py --example basic       # Basic TTS demo
python example_tts.py --example emotion     # Emotion control demo
python example_tts.py --example scientific  # Scientific text examples
python example_tts.py --example batch       # Batch processing
python example_tts.py --example all         # Run all examples

# Voice cloning (requires a voice sample)
python example_tts.py --example voice --voice-sample path/to/voice.wav

# Test TTS functionality
python test_tts.py
```

**Key TTS Features:**
- 🎭 **Emotion Control**: Adjust speech emotion intensity (excited, calm, dramatic, etc.)
- 🎤 **Voice Cloning**: Use any voice sample to synthesize speech
- ⚡ **Fast Inference**: Sub-200ms latency for real-time applications
- 🛡️ **Watermarking**: Built-in PerTh neural watermarking for responsible AI
- 🔊 **Auto-playback**: Automatic audio playback on macOS, Linux, and Windows

## 🔄 Migration from Original Code

### Mapping of Original Files to New Architecture

| Original File | New Location | Purpose |
|--------------|--------------|---------|
| `transformer_test.py` | `main.py test` command | Model testing |
| `build_index.py` | `services/document_indexer.py` + `main.py build-index` | PDF indexing |
| `rag_chat.py` | Split across multiple modules: |  |
| - Model loading | `core/model_manager.py` | Model management |
| - Vector store | `core/vector_store.py` | Vector operations |
| - Retrieval | `core/retriever.py` | Document retrieval |
| - RAG logic | `services/rag_service.py` | Q&A service |
| - CLI interface | `main.py chat` command | User interface |

### Running Original Scripts vs Refactored

```bash
# Original
python build_index.py
python rag_chat.py
python transformer_test.py

# Refactored
python main.py build-index
python main.py chat
python main.py test
```

## 🎯 Key Improvements

### 1. **Separation of Concerns**
- Core business logic separated from UI
- Clear layer boundaries (Core → Services → UI)
- Each module has a single responsibility

### 2. **Configuration Management**
- Centralized YAML configuration
- No hardcoded values in code
- Easy environment-specific settings

### 3. **Error Handling & Logging**
- Comprehensive error handling
- Structured logging throughout
- Configurable log levels and outputs

### 4. **Testability**
- Unit tests for core modules
- Dependency injection for easy mocking
- Clear interfaces between components

### 5. **Extensibility**
- Easy to add new features
- Support for multiple UI types (CLI, API)
- Plugin-style architecture

### 6. **Resource Management**
- Proper cleanup of resources
- Memory optimization
- Efficient model loading

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_core.py

# Run with coverage
python -m pytest --cov=core --cov=services tests/
```

## 📝 API Examples

### Using as a Library

```python
from utils import load_config
from core import ModelManager, EmbeddingsManager, VectorStoreManager, DocumentRetriever
from services import RAGService

# Load configuration
config = load_config("config/settings.yaml")

# Initialize components
model_manager = ModelManager(config)
model_manager.load_model()

embeddings_manager = EmbeddingsManager(config)
embeddings = embeddings_manager.load_embeddings()

vector_store_manager = VectorStoreManager(config)
vector_store_manager.load_vector_store(embeddings)

retriever = DocumentRetriever(vector_store_manager, config)

# Create RAG service
rag_service = RAGService(model_manager, retriever, config)

# Get answer
result = rag_service.answer_question(
    "What is biomechanics?",
    return_context=True
)

print(result["answer"])
```

### Using TTS as a Library

```python
from utils import load_config
from services import TTSService, EmotionalTTSService

# Load configuration
config = load_config("config/settings.yaml")

# Basic TTS
tts = TTSService(config)
tts.load_model()

# Generate speech
audio_path = tts.synthesize("Hello from the RAG system!")
print(f"Audio saved to: {audio_path}")

# With emotion control
emotional_tts = EmotionalTTSService(config)
emotional_tts.load_model()

# Generate with emotion preset
audio_path = emotional_tts.synthesize_with_emotion(
    "This is an exciting discovery!",
    emotion="excited"
)

# Voice cloning
cloned_audio = tts.convert_voice(
    "This uses your voice",
    voice_sample_path="path/to/voice.wav"
)

# Cleanup
tts.cleanup()
```

## 🔧 Customization

### Adding New Models

1. Update `config/settings.yaml` with new model ID
2. Optionally extend `ModelManager` for model-specific logic

### Custom Embeddings

1. Update embeddings configuration
2. Extend `EmbeddingsManager` if needed

### Different Vector Stores

1. Create new vector store manager implementing same interface
2. Update configuration to use new store

## 📊 Performance Considerations

- **GPU Memory**: Use CPU for embeddings to save GPU memory
- **Batch Processing**: Use `answer_batch()` for multiple questions
- **Index Updates**: Use `update_index()` for incremental updates
- **Resource Cleanup**: Always call cleanup methods when done

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use CPU for embeddings
2. **Slow Loading**: First run downloads models, subsequent runs are faster
3. **Index Not Found**: Run `build-index` command first
4. **Import Errors**: Ensure all dependencies are installed

## 📄 License

Same as original project

## 🤝 Contributing

1. Follow the modular architecture
2. Add tests for new features
3. Update configuration schema if needed
4. Document new modules and functions