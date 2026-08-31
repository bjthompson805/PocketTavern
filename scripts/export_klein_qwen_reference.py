#!/usr/bin/env python3
"""Create a reference Qwen3 context tensor for FLUX.2 [klein] 4B.

The official Klein transformer expects the concatenation of Qwen hidden states 9, 18, and 27:
[1, 512, 3 * 2560] = [1, 512, 7680].  This script uses the downloaded official shards and
tokenizer, then writes inputs/outputs suitable for native denoiser validation.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_ROOT = Path("/home/brandont/Downloads/flux2_klein_qwen")
DEFAULT_OUTPUT = Path("/home/brandont/code/litert-torch/scratch/models/klein_qwen_reference")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default="a cinematic photograph of a small red fox in a moonlit forest")
    args = parser.parse_args()
    text_encoder = args.root / "text_encoder"
    tokenizer_dir = args.root / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        text_encoder, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="cpu",
    ).eval()
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(chat, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.inference_mode():
        output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                       output_hidden_states=True, use_cache=False)
    context = torch.cat([output.hidden_states[layer] for layer in (9, 18, 27)], dim=-1).float().contiguous()
    if context.shape != (1, 512, 7680) or not torch.isfinite(context).all():
        raise RuntimeError(f"invalid context tensor: shape={tuple(context.shape)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context.numpy().tofile(args.output_dir / "context.bin")
    inputs.input_ids.numpy().astype("int32").tofile(args.output_dir / "input_ids.i32.bin")
    inputs.attention_mask.numpy().astype("int32").tofile(args.output_dir / "attention_mask.i32.bin")
    (args.output_dir / "prompt.txt").write_text(args.prompt + "\n")
    print(f"wrote context.bin: {context.numel() * 4} bytes; token_count={inputs.attention_mask.sum().item()}")


if __name__ == "__main__":
    main()
