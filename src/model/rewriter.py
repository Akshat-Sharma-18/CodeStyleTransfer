"""Wrap a model as the same `(code, direction) -> code` callable the eval
harness scores every baseline with.

Two modes, and the comparison between them is the result that justifies
training at all (the spec's v2 bar):

  base      the pretrained backbone, prompted but never fine-tuned
  adapter   the same backbone with the LoRA adapter loaded

Because both go through identical decoding and identical scoring, a difference
in the eval table is attributable to the fine-tune rather than to prompt or
harness differences.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_MODEL = "Salesforce/codet5p-220m"

STYLE_TOKEN = {"to_terse": "<to_terse>", "to_verbose": "<to_verbose>"}


class ModelRewriter:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        adapter_path: str | None = None,
        max_source: int = 512,
        max_new_tokens: int = 512,
        num_beams: int = 4,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        self.model = model.to(self.device).eval()
        self.max_source = max_source
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def __call__(self, code: str, direction: str) -> str:
        prompt = f"{STYLE_TOKEN[direction]}\n{code}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_source,
            truncation=True,
        ).to(self.device)

        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
