#!/usr/bin/env bash
# Stage compiled FLUX.2 [klein] NPU artifacts into one flat directory for adb push.
# Usage: stage_klein_npu_artifacts.sh <transformer_dir> <output_dir> [component_dir]
#
# transformer_dir is the learned block-artifact root.  component_dir is normally
# scratch/models/flux2_klein_components and contributes the input/modulation,
# final-layer, and split-VAE artifacts.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <transformer_dir> <output_dir> [component_dir]" >&2
  exit 2
fi

transformer_dir=$1
output_dir=$2
component_dir=${3:-}
if [[ ! -d $transformer_dir ]]; then
  echo "Transformer directory does not exist: $transformer_dir" >&2
  exit 2
fi
if [[ -n $component_dir && ! -d $component_dir ]]; then
  echo "Component directory does not exist: $component_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"

# The exporter writes Google Tensor compilations below *_noflags_aot directories.  Their
# basenames contain the block number, so a flat destination retains unambiguous lookups.
stage_source() {
  local source_dir=$1
  while IFS= read -r -d '' model; do
    cp -f "$model" "$output_dir/${model##*/}"
  done < <(find "$source_dir" -type f -path '*_noflags_aot/*_Google_Tensor_G5.tflite' -print0)
}

stage_source "$transformer_dir"
if [[ -n $component_dir ]]; then
  stage_source "$component_dir"
  # Native Q/K RMSNorm scale vectors (klein_qk_norm_rope.hpp) -- KleinDoubleBlockEngine::Load()
  # expects them under a qk_norm_scales/ subdirectory of the pushed model dir, not flattened.
  if [[ -d "$component_dir/qk_norm_scales" ]]; then
    mkdir -p "$output_dir/qk_norm_scales"
    cp -f "$component_dir"/qk_norm_scales/*.bin "$output_dir/qk_norm_scales/"
  fi
fi

echo "Staged $(find "$output_dir" -maxdepth 1 -type f -name '*_Google_Tensor_G5.tflite' | wc -l) artifacts in $output_dir"
if [[ -d "$output_dir/qk_norm_scales" ]]; then
  echo "Staged $(find "$output_dir/qk_norm_scales" -type f -name '*.bin' | wc -l) Q/K norm-scale files in $output_dir/qk_norm_scales"
fi
