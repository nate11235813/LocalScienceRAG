# Supported Models

The application supports standard Hugging Face causal language models as well as
`GPT-OSS` checkpoints.  GPT‑OSS models are Llama‑compatible weights released by
Open Source Systems and require a few extra steps when loading.

## GPT‑OSS

* The loader registers the custom `gpt_oss` architecture with the Transformers
  library and treats it as a Llama variant.
* Tokenizers are patched to ensure `bos_token`, `eos_token`, and `pad_token` are
  set and include a minimal chat formatting template.
* Model weights are loaded using a fallback strategy:
  1. load with `trust_remote_code=True`
  2. retry forcing the Llama architecture
  3. finally attempt a generic `AutoModelForCausalLM` load

To add a new GPT‑OSS variant simply reference its model id in the configuration;
the loader is selected automatically whenever the model id contains
`"gpt-oss"`.
