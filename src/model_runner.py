import gc
import os
from typing import Optional, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class HuggingFaceRunner:
    def __init__(self, model_id: str, device: str = "auto"):
        if torch is None or AutoModelForCausalLM is None:
            raise ImportError(
                "PyTorch and Transformers are required. Run 'pip install -r requirements.txt'."
            )

        self.model_id = model_id
        self.device = self._resolve_device(device)

        # Load tokenizer with pad token fallback
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model: utilize device_map='auto' on CUDA for large models (>1B, 13B+)
        model_kwargs = {"low_cpu_mem_usage": True}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def _resolve_device(self, preferred: str) -> str:
        if preferred == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if preferred == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return preferred

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
    ) -> Tuple[str, int]:
        target_device = getattr(self.model, "device", self.device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(target_device)
        prompt_len = inputs.input_ids.shape[1]

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **generate_kwargs)

        # Slice to keep only newly generated tokens
        new_tokens = output_tokens[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        token_count = int(new_tokens.shape[0])

        return text, token_count

    def cleanup(self) -> None:
        """Release weights and purge CUDA/RAM buffers to avoid OOM in sequential runs."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
