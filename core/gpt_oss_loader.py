"""Utilities for loading GPT-OSS models.

This module registers the ``gpt_oss`` architecture as Llama compatible and
provides a loader that fixes tokenizer metadata and attempts multiple strategies
for weight loading.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
)
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

logger = logging.getLogger(__name__)

CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}{{ message['content'] }}"
    "{%- elif message['role'] == 'user' -%}User: {{ message['content'] }}"
    "{%- elif message['role'] == 'assistant' -%}Assistant: {{ message['content'] }}"
    "{%- endif -%}"
    "{%- if not loop.last %}\n{% endif -%}"
    "{%- endfor -%}"
    "{%- if messages[-1]['role'] != 'assistant' %}Assistant: {% endif -%}"
)


def _register_architecture() -> None:
    """Register GPT-OSS as a Llama-compatible architecture."""
    try:
        if "gpt_oss" not in CONFIG_MAPPING:
            CONFIG_MAPPING.register("gpt_oss", LlamaConfig)
        if LlamaConfig not in MODEL_FOR_CAUSAL_LM_MAPPING:
            MODEL_FOR_CAUSAL_LM_MAPPING.register(LlamaConfig, LlamaForCausalLM)
        logger.debug("Registered gpt_oss architecture")
    except Exception as exc:  # pragma: no cover - registration is best-effort
        logger.debug("Architecture registration skipped: %s", exc)


_register_architecture()


class GPTOSSLoader:
    """Loader for GPT-OSS models."""

    @staticmethod
    def load_model_and_tokenizer(
        model_id: str,
        *,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a GPT-OSS model and tokenizer.

        The loader first fixes tokenizer special tokens and provides a minimal
        chat template. It then attempts the following loading strategies:

        1. Load with ``trust_remote_code=True``.
        2. Force loading as a Llama model.
        3. Generic ``AutoModelForCausalLM`` loading.

        Args:
            model_id: Hugging Face model identifier.
            device: Device to place the model on.
            dtype: Optional ``torch.dtype`` for model weights.
            **kwargs: Extra arguments passed to ``from_pretrained``.

        Returns:
            Tuple of (model, tokenizer).
        """

        logger.info("Loading GPT-OSS model %s", model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({"bos_token": "<s>"})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            tokenizer.chat_template = CHAT_TEMPLATE

        device_map = None if device == "cpu" else device
        errors = []

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=True,
                **kwargs,
            )
        except Exception as err1:  # pragma: no cover - exercised in tests via mock
            errors.append(f"trust_remote_code failed: {err1}")
            try:
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
                if getattr(config, "model_type", "") == "gpt_oss":
                    config.model_type = "llama"
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    device_map=device_map,
                    torch_dtype=dtype,
                    trust_remote_code=False,
                    **kwargs,
                )
            except Exception as err2:  # pragma: no cover
                errors.append(f"llama fallback failed: {err2}")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map=device_map,
                        torch_dtype=dtype,
                        trust_remote_code=False,
                        **kwargs,
                    )
                except Exception as err3:
                    errors.append(f"generic loading failed: {err3}")
                    msg = " | ".join(errors)
                    raise RuntimeError(
                        f"Unable to load GPT-OSS model {model_id}: {msg}"
                    ) from err3

        return model, tokenizer
