from functools import cached_property
import os

import numpy as np

from rag.corpus import MODEL_CACHE


EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GENERATION_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def torch_device():
    import torch

    return torch.device(os.environ.get("RAG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))


class Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL, revision=None):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch_device()
        directory = os.environ.get("RAG_MODEL_CACHE", str(MODEL_CACHE))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=directory)
        self.model = AutoModel.from_pretrained(model_name, revision=revision, cache_dir=directory)
        self.model.to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def encode(self, texts, *, query=False, batch_size=32):
        import torch

        prefix = "query: " if query else "passage: "
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    [prefix + text for text in texts[start:start + batch_size]],
                    padding=True, truncation=True, max_length=512, return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch.attention_mask.unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled.float()).cpu().numpy())
        return np.concatenate(batches)


class LocalGenerator:
    def __init__(self, model_name):
        self.model_name = model_name

    @cached_property
    def runtime(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch_device()
        directory = os.environ.get("RAG_MODEL_CACHE", str(MODEL_CACHE))
        revision = os.environ.get("RAG_GENERATION_REVISION") or (
            GENERATION_REVISION if self.model_name == GENERATION_MODEL else None
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=directory, revision=revision)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, cache_dir=directory, revision=revision, dtype=dtype,
        ).to(device).eval()
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, model

    def prepare(self):
        _ = self.runtime

    def __call__(self, messages, max_tokens):
        import torch

        tokenizer, model = self.runtime
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


class ChatEndpoint:
    def __init__(self, base_url, model_name, key):
        import requests

        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model_name = model_name
        self.session = requests.Session()
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"

    def prepare(self):
        pass

    def __call__(self, messages, max_tokens):
        response = self.session.post(self.url, json={
            "model": self.model_name, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
        }, timeout=(10, 180))
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("The generation endpoint did not return text")
        return content.strip()


def configured_generator():
    if base_url := os.environ.get("RAG_GENERATION_BASE_URL"):
        model = os.environ.get("RAG_GENERATION_MODEL")
        if not model:
            raise ValueError("Set RAG_GENERATION_MODEL for the configured generation endpoint")
        return ChatEndpoint(base_url, model, os.environ.get("RAG_GENERATION_API_KEY"))
    return LocalGenerator(os.environ.get("RAG_GENERATION_MODEL", GENERATION_MODEL))
