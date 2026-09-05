"""Load the CodeT5 tokenizer despite its legacy config.

Every CodeT5 / CodeT5+ checkpoint still ships `tokenizer_config.json` with the
2021-era `{"content": "<s>", "__type": "AddedToken"}` serialization. transformers
5.x no longer rehydrates those dicts into `AddedToken` objects before handing
them to `add_tokens`, so `AutoTokenizer.from_pretrained` dies with:

    TypeError: Input must be a List[Union[str, AddedToken]]

Both the fast and slow paths fail, there is no `tokenizer.json` in the repo to
load directly, and the fast-conversion fallback needs to instantiate the slow
tokenizer first -- which is the thing that crashes. (The error it reports,
"You need to have sentencepiece or tiktoken installed", is misleading:
sentencepiece is installed and irrelevant, since this is a byte-level BPE.)

The way through is to skip transformers' conversion entirely and build the
backend straight from `vocab.json` + `merges.txt` with the `tokenizers`
library, which is all a byte-level BPE actually needs. Verified to round-trip
Python source exactly.

The alternative was pinning transformers to a 4.x release for the whole
project, or abandoning the spec's recommended encoder-decoder for a
decoder-only model with a modern tokenizer. This keeps both the backbone and a
current transformers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "sep_token": "</s>",
    "cls_token": "<s>",
    "mask_token": "<mask>",
}


def load_codet5_tokenizer(model_name: str, model_max_length: int = 512):
    """Build a working fast tokenizer for a CodeT5-family checkpoint.

    The post-processor is not optional decoration. A bare byte-level BPE
    appends nothing, so `tokenizer(target)["input_ids"]` comes back with no
    `</s>` -- which means seq2seq training labels never contain an EOS and the
    model is never taught where to stop. The first fine-tune here did exactly
    that: it generated until `max_new_tokens` and emitted the same function
    six times over, scoring 45% parse rate and 0.595 content. Restoring the
    `$A </s>` template is what `AutoTokenizer` would have done for us.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab_path = hf_hub_download(model_name, "vocab.json")
    merges_path = hf_hub_download(model_name, "merges.txt")

    backend = ByteLevelBPETokenizer(vocab_path, merges_path)

    eos = SPECIAL_TOKENS["eos_token"]
    eos_id = backend.token_to_id(eos)
    if eos_id is None:
        raise RuntimeError(f"{model_name} vocab has no {eos} token to terminate sequences with")
    backend.post_processor = TemplateProcessing(
        single=f"$A {eos}",
        pair=f"$A {eos} $B {eos}",
        special_tokens=[(eos, eos_id)],
    )

    serialized = Path(tempfile.gettempdir()) / f"{model_name.replace('/', '_')}_tokenizer.json"
    backend.save(str(serialized))

    return PreTrainedTokenizerFast(
        tokenizer_file=str(serialized),
        model_max_length=model_max_length,
        **SPECIAL_TOKENS,
    )


def load_tokenizer(model_name: str, model_max_length: int = 512):
    """AutoTokenizer where it works, the CodeT5 workaround where it doesn't."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except TypeError:
        return load_codet5_tokenizer(model_name, model_max_length)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
