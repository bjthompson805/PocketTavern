#!/usr/bin/env python3
"""Export the official FLUX.2 autoencoder decoder at the 1024px Klein shape.

Input is the transformer's final [1, 4096, 128] prediction reshaped to [1, 128, 64, 64].
The wrapper applies the autoencoder's stored inverse batch-normalization and 2x2 latent unpacking
before running the 32-channel Flux2 decoder, producing [1, 3, 1024, 1024].
"""
import argparse
import sys
from pathlib import Path

import litert_torch
import torch
from einops import rearrange
from safetensors.torch import load_file


DEFAULT_AE = Path("/home/brandont/Downloads/ae.safetensors")
DEFAULT_UPSTREAM = Path("/home/brandont/code/litert-torch/scratch/flux2_upstream/flux2/src")
DEFAULT_OUTPUT = Path("/home/brandont/code/litert-torch/scratch/models/flux2_klein_components")


class Flux2DecoderPreMid(torch.nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.decoder = autoencoder.decoder
        self.register_buffer("running_mean", autoencoder.bn.running_mean)
        self.register_buffer("running_var", autoencoder.bn.running_var)
        self.eps = autoencoder.bn_eps

    def forward(self, packed_latents):
        # Match AutoEncoder.decode(): [128, 64, 64] -> [32, 128, 128] -> RGB 1024px.
        std = torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)
        latents = packed_latents * std + self.running_mean.view(1, -1, 1, 1)
        latents = rearrange(latents, "b (c pi pj) h w -> b c (h pi) (w pj)", pi=2, pj=2)
        hidden = self.decoder.post_quant_conv(latents)
        hidden = self.decoder.conv_in(hidden)
        hidden = self.decoder.mid.block_1(hidden)
        hidden = self.decoder.mid.attn_1(hidden)
        return self.decoder.mid.block_2(hidden)


class Flux2DecoderUpLevel(torch.nn.Module):
    def __init__(self, decoder, level, final):
        super().__init__()
        self.level = decoder.up[level]
        self.final = final
        self.norm_out = decoder.norm_out if final else None
        self.conv_out = decoder.conv_out if final else None

    def forward(self, hidden):
        for block in self.level.block:
            hidden = block(hidden)
        if self.final:
            return self.conv_out(torch.nn.functional.silu(self.norm_out(hidden)))
        return self.level.upsample(hidden)


class Flux2DecoderResidualBlock(torch.nn.Module):
    def __init__(self, decoder, block_index):
        super().__init__()
        self.block = decoder.up[0].block[block_index]

    def forward(self, hidden):
        return self.block(hidden)


class Flux2DecoderRgbHead(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.norm_out = decoder.norm_out
        self.conv_out = decoder.conv_out

    def forward(self, hidden):
        return self.conv_out(torch.nn.functional.silu(self.norm_out(hidden)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae", type=Path, default=DEFAULT_AE)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=(
        "pre_mid", "up_3", "up_2", "up_1", "up_0_final",
        "up_0_block_0", "up_0_block_1", "up_0_block_2", "up_0_head",
    ), required=True)
    args = parser.parse_args()
    if not args.ae.is_file() or not args.upstream.is_dir():
        raise FileNotFoundError("autoencoder weights or Flux2 upstream source is unavailable")
    sys.path.insert(0, str(args.upstream))
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    with torch.device("meta"):
        autoencoder = AutoEncoder(AutoEncoderParams())
    autoencoder.load_state_dict(load_file(str(args.ae)), strict=True, assign=True)
    if args.stage == "pre_mid":
        wrapper = Flux2DecoderPreMid(autoencoder)
        sample = (torch.randn(1, 128, 64, 64),)
    elif args.stage.startswith("up_0_block_"):
        block_index = int(args.stage[-1])
        channels = 256 if block_index == 0 else 128
        wrapper = Flux2DecoderResidualBlock(autoencoder.decoder, block_index)
        sample = (torch.randn(1, channels, 1024, 1024),)
    elif args.stage == "up_0_head":
        wrapper = Flux2DecoderRgbHead(autoencoder.decoder)
        sample = (torch.randn(1, 128, 1024, 1024),)
    else:
        level = int(args.stage[3])
        final = level == 0
        # Native decoder order is up.3 -> up.2 -> up.1 -> up.0.
        channel_and_size = {3: (512, 128), 2: (512, 256), 1: (512, 512), 0: (256, 1024)}
        channels, size = channel_and_size[level]
        wrapper = Flux2DecoderUpLevel(autoencoder.decoder, level, final)
        sample = (torch.randn(1, channels, size, size),)
    wrapper = wrapper.float().eval()
    with torch.inference_mode():
        print("reference output:", tuple(wrapper(*sample).shape), flush=True)
    edge_model = litert_torch.convert(wrapper, sample_args=sample, lightweight_conversion=True,
                                      runtime_constant_folding=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"vae_decoder_{args.stage}.tflite"
    edge_model.export(str(output))
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
