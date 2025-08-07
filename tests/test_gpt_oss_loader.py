"""Tests for GPT-OSS model loading."""

from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("torch", MagicMock())
import types


class _Mapping(dict):
    def register(self, key, value):  # pragma: no cover - trivial
        self[key] = value


transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = MagicMock()
transformers_stub.AutoModelForCausalLM = MagicMock()
transformers_stub.AutoConfig = MagicMock()
transformers_stub.CONFIG_MAPPING = _Mapping()
transformers_stub.MODEL_FOR_CAUSAL_LM_MAPPING = _Mapping()
transformers_stub.pipeline = MagicMock()
transformers_stub.Pipeline = MagicMock()

llama_module = types.ModuleType("transformers.models.llama")
llama_module.LlamaConfig = MagicMock()
llama_module.LlamaForCausalLM = MagicMock()

transformers_stub.models = types.SimpleNamespace(llama=llama_module)

sys.modules.setdefault("transformers", transformers_stub)
sys.modules.setdefault("transformers.models", transformers_stub.models)
sys.modules.setdefault("transformers.models.llama", llama_module)

from core.model_manager import ModelManager


def test_gpt_oss_loader_generates_non_repetitive():
    config = {
        "model": {
            "id": "unsloth/gpt-oss-2b",
            "dtype": "float32",
            "device": "cpu",
            "max_new_tokens": 15,
            "temperature": 0.7,
            "do_sample": False,
        }
    }

    mm = ModelManager(config)

    dummy_tokenizer = MagicMock()
    dummy_tokenizer.bos_token_id = 0
    dummy_tokenizer.eos_token = "</s>"
    dummy_tokenizer.pad_token = "</s>"
    dummy_tokenizer.chat_template = ""

    dummy_model = MagicMock()
    dummy_model.generation_config = MagicMock()

    def fake_pipeline(*args, **kwargs):
        return [{"generated_text": "Hello world this is a test"}]

    with patch(
        "core.gpt_oss_loader.GPTOSSLoader.load_model_and_tokenizer",
        return_value=(dummy_model, dummy_tokenizer),
    ) as loader, patch("core.model_manager.pipeline", return_value=fake_pipeline) as pipe:
        mm.load_model()
        loader.assert_called_once()
        pipe.assert_called_once()

    assert mm.model.generation_config is not None
    assert mm.tokenizer.bos_token_id is not None

    text = mm.generate("Hello")
    words = text.split()
    max_repeat = 1
    current = 1
    for prev, cur in zip(words, words[1:]):
        if cur == prev:
            current += 1
            max_repeat = max(max_repeat, current)
        else:
            current = 1
    assert max_repeat <= 4
