"""LoRA fine-tune of a small code model for style transfer.

Backbone: Qwen2.5-Coder-0.5B, decoder-only.

The spec suggests CodeT5/CodeT5+ (encoder-decoder) first, and that was the
original choice here. It does not work on this machine: every CodeT5 tokenizer
still ships the legacy `{"content": ..., "__type": "AddedToken"}` config form,
which transformers 5.16 no longer converts before calling `add_tokens`, so
loading dies with `TypeError: Input must be a List[Union[str, AddedToken]]`.
Slow and fast paths both fail, and there is no `tokenizer.json` to bypass it
with. Pinning an old transformers release to rescue a 2021 checkpoint is worse
than taking the spec's other sanctioned option -- decoder-only -- with a model
whose tokenizer loads.

Both directions live in one model. The style token (`<to_terse>` /
`<to_verbose>`) is the first line of every input, so direction is part of the
prompt rather than a separate head. The tokens are left as ordinary subwords
rather than registered as special: adding them resizes the embedding matrix,
which then has to be trained alongside the adapters. Revisit only if the model
turns out to ignore the direction.

Loss is masked to the completion only -- the model is scored on the rewrite it
produces, never on reciting the prompt back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-0.5B"

PROMPT_TEMPLATE = "# rewrite the following python function\n{style}\n{code}\n# rewritten:\n"


def load_split(split: str) -> list[dict]:
    path = ROOT / "data" / "pairs" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run src/transforms/generate_pairs.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_prompt(style_token: str, code: str) -> str:
    return PROMPT_TEMPLATE.format(style=style_token, code=code)


def split_input(example: dict) -> tuple[str, str]:
    """The dataset stores '<to_terse>\\n<code>'; separate the two."""
    head, _, body = example["input"].partition("\n")
    return head.strip(), body


class PairDataset:
    """Prompt + completion, with the prompt masked out of the loss."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        style_token, code = split_input(example)
        prompt = format_prompt(style_token, code)
        completion = example["target"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + completion_ids)[: self.max_length]
        # -100 tells the loss to ignore the prompt tokens.
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output", default=str(ROOT / "data" / "checkpoints" / "lora"))
    parser.add_argument("--limit", type=int, default=None, help="truncate the training set (smoke tests)")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    train_examples = load_split("train")
    val_examples = load_split("val")
    if args.limit:
        train_examples = train_examples[: args.limit]
        val_examples = val_examples[: max(8, args.limit // 8)]
    print(f"train {len(train_examples)} | val {len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=use_cuda,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=PairDataset(train_examples, tokenizer, args.max_length),
        eval_dataset=PairDataset(val_examples, tokenizer, args.max_length),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100),
    )

    trainer.train()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nsaved adapter to {out_dir}")


if __name__ == "__main__":
    main()
