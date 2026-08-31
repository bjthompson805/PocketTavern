"""Extracts per-head Q/K RMSNorm learned scale vectors ([128] each) from the merged Klein
checkpoint, for every double-stream block's img/txt attention and every single-stream block's
attention, and writes each as a raw little-endian float32 .bin file.

These scales are the "risky tail" that Tensor G5's compiled fused-QKV artifact gets wrong (see
docs/FLUX2_KLEIN_HANDOFF.md, "Critical current status"). The repair moves Q/K RMSNorm + RoPE to
native C++, which needs these learned scale vectors as plain host-readable data -- the NPU no
longer applies them.

Usage:
  .venv/bin/python scripts/export_klein_qk_norm_scales.py [--output-dir DIR]
"""

import argparse
from pathlib import Path

from safetensors import safe_open

CHECKPOINT = Path("/home/brandont/Downloads/unstableRevolutionF2K_AlphaF2K4BFp16.safetensors")
PREFIX = "model.diffusion_model."
DEFAULT_OUTPUT_DIR = Path(
    "/home/brandont/code/litert-torch/scratch/models/flux2_klein_components/qk_norm_scales"
)
NUM_DOUBLE_BLOCKS = 5
NUM_SINGLE_BLOCKS = 20


def write_scale(f, key: str, out_path: Path) -> None:
    tensor = f.get_tensor(key).float()
    assert tensor.shape == (128,), f"{key}: unexpected shape {tuple(tensor.shape)}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor.numpy().tofile(out_path)
    print(f"wrote {out_path} ({tensor.numel()} floats) from {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    with safe_open(args.checkpoint, "pt") as f:
        for i in range(NUM_DOUBLE_BLOCKS):
            for stream in ("img", "txt"):
                for norm in ("query_norm", "key_norm"):
                    key = f"{PREFIX}double_blocks.{i}.{stream}_attn.norm.{norm}.scale"
                    out = args.output_dir / f"double{i}_{stream}_{norm}.bin"
                    write_scale(f, key, out)
        for i in range(NUM_SINGLE_BLOCKS):
            for norm in ("query_norm", "key_norm"):
                key = f"{PREFIX}single_blocks.{i}.norm.{norm}.scale"
                out = args.output_dir / f"single{i}_{norm}.bin"
                write_scale(f, key, out)

    total = 2 * 2 * NUM_DOUBLE_BLOCKS + 2 * NUM_SINGLE_BLOCKS
    print(f"done: {total} scale files in {args.output_dir}")


if __name__ == "__main__":
    main()
