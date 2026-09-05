"""LoRA fine-tune of a small code model for style transfer.

Backbone: CodeT5+ 220M, an encoder-decoder, as the spec recommends. Seq2seq is
the natural shape here -- the task is literally "this code in, that code out"
-- and at 220M it fits an 8GB card with room for a real batch size.

Its tokenizer needs a workaround to load at all under transformers 5.x; see
`tokenizer_compat.py` for why.

Both directions live in one model. The style token (`<to_terse>` /
`<to_verbose>`) is the first line of every input, so the direction is part of
the prompt rather than a separate head. The tokens are left as ordinary
subwords rather than registered as special: adding them resizes the embedding
matrix, which then has to be trained alongside the adapters. Revisit only if
the model turns out to ignore the direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model.tokenizer_compat import load_tokenizer  # noqa: E402

DEFAULT_MODEL = "Salesforce/codet5p-220m"


def load_split(split: str) -> list[dict]:
    path = ROOT / "data" / "pairs" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run src/transforms/generate_pairs.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class PairDataset:
    def __init__(self, examples: list[dict], tokenizer, max_source: int, max_target: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source = max_source
        self.max_target = max_target

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        model_inputs = self.tokenizer(example["input"], max_length=self.max_source, truncation=True)
        labels = self.tokenizer(example["target"], max_length=self.max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-source", type=int, default=512)
    parser.add_argument("--max-target", type=int, default=512)
    parser.add_argument("--output", default=str(ROOT / "data" / "checkpoints" / "lora"))
    parser.add_argument("--limit", type=int, default=None, help="truncate the training set (smoke tests)")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    train_examples = load_split("train")
    val_examples = load_split("val")
    if args.limit:
        train_examples = train_examples[: args.limit]
        val_examples = val_examples[: max(8, args.limit // 8)]
    print(f"train {len(train_examples)} | val {len(val_examples)}")

    tokenizer = load_tokenizer(args.model, model_max_length=args.max_source)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    use_cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
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
        fp16=use_cuda,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=PairDataset(train_examples, tokenizer, args.max_source, args.max_target),
        eval_dataset=PairDataset(val_examples, tokenizer, args.max_source, args.max_target),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest"),
    )

    trainer.train()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nsaved adapter to {out_dir}")


if __name__ == "__main__":
    main()
