#!/usr/bin/env python3
"""Export FLUX.2 [klein] checkpoint-backed non-block denoiser components to LiteRT.

The chunked transformer block exports intentionally exclude these small-but-essential pieces:
image/context projections, time MLP, modulation projections, and final layer.  This script loads
only their tensors from the merged checkpoint and emits fixed 1024px/512-token artifacts.  It does
not export Qwen3 or the Flux autoencoder: neither is contained in that checkpoint.
"""
import argparse
import sys
from pathlib import Path

import litert_torch
import torch
import torch.nn.functional as F
from safetensors import safe_open


DEFAULT_CHECKPOINT = Path("/home/brandont/Downloads/unstableRevolutionF2K_AlphaF2K4BFp16.safetensors")
DEFAULT_UPSTREAM = Path("/home/brandont/code/litert-torch/scratch/flux2_upstream/flux2/src")
DEFAULT_OUTPUT = Path("/home/brandont/code/litert-torch/scratch/models/flux2_klein_components")
HIDDEN = 3072


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--component", choices=("all", "img_in", "txt_in", "time_in", "mod_img", "mod_txt", "mod_single", "final"), default="all")
    return parser.parse_args()


def tensors(checkpoint, prefix):
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        return {key.removeprefix(prefix): source.get_tensor(key).float() for key in source.keys() if key.startswith(prefix)}


class Linear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        return F.linear(x, self.weight)


class TimeIn(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.in_weight = torch.nn.Parameter(state["in_layer.weight"])
        self.out_weight = torch.nn.Parameter(state["out_layer.weight"])

    def forward(self, timestep_embedding):
        return F.linear(F.silu(F.linear(timestep_embedding, self.in_weight)), self.out_weight)


class Modulation(torch.nn.Module):
    def __init__(self, weight, multiplier):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.multiplier = multiplier

    def forward(self, vec):
        return F.linear(F.silu(vec), self.weight).chunk(self.multiplier, dim=-1)


class FinalLayer(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.linear_weight = torch.nn.Parameter(state["linear.weight"])
        self.mod_weight = torch.nn.Parameter(state["adaLN_modulation.1.weight"])

    def forward(self, hidden, vec):
        shift, scale = F.linear(F.silu(vec), self.mod_weight).chunk(2, dim=-1)
        hidden = (1 + scale[:, None, :]) * F.layer_norm(hidden, (HIDDEN,), None, None, 1e-6) + shift[:, None, :]
        return F.linear(hidden, self.linear_weight)


def export(name, module, sample_args, output_dir):
    module.eval()
    with torch.inference_mode():
        outputs = module(*sample_args)
    shapes = [tuple(x.shape) for x in outputs] if isinstance(outputs, tuple) else [tuple(outputs.shape)]
    print(f"{name}: outputs={shapes}", flush=True)
    edge_model = litert_torch.convert(module, sample_args=sample_args, lightweight_conversion=True, runtime_constant_folding=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output_dir / f"{name}.tflite"))


def main():
    args = parse_args()
    if not args.checkpoint.is_file() or not args.upstream.is_dir():
        raise FileNotFoundError("checkpoint or upstream source is unavailable")
    sys.path.insert(0, str(args.upstream))
    wanted = lambda name: args.component in ("all", name)
    root = "model.diffusion_model."
    if wanted("img_in"):
        export("img_in", Linear(tensors(args.checkpoint, root + "img_in.")["weight"]), (torch.randn(1, 4096, 128),), args.output_dir)
    if wanted("txt_in"):
        export("txt_in", Linear(tensors(args.checkpoint, root + "txt_in.")["weight"]), (torch.randn(1, 512, 7680),), args.output_dir)
    if wanted("time_in"):
        export("time_in", TimeIn(tensors(args.checkpoint, root + "time_in.")), (torch.randn(1, 256),), args.output_dir)
    for name, prefix, multiplier in (("mod_img", "double_stream_modulation_img.", 6), ("mod_txt", "double_stream_modulation_txt.", 6), ("mod_single", "single_stream_modulation.", 3)):
        if wanted(name):
            export(name, Modulation(tensors(args.checkpoint, root + prefix)["lin.weight"], multiplier), (torch.randn(1, HIDDEN),), args.output_dir)
    if wanted("final"):
        export("final", FinalLayer(tensors(args.checkpoint, root + "final_layer.")), (torch.randn(1, 4096, HIDDEN), torch.randn(1, HIDDEN)), args.output_dir)


if __name__ == "__main__":
    main()
