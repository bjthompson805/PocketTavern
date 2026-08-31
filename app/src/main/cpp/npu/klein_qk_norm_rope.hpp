// Native Q/K per-head RMSNorm + FLUX RoPE application for FLUX.2 [klein].
//
// Tensor G5's compiled fused-QKV artifact returns near-zero Q/K under real production
// activations (V is correct) -- see docs/FLUX2_KLEIN_HANDOFF.md, "Critical current status".
// The repair keeps the expensive learned QKV projection matrix on NPU (the "raw_qkv" artifacts)
// and performs this small, cheap per-head tail natively instead of trusting the NPU-compiled
// version of it.
//
// Deliberately header-only and free of any LiteRT/JNI/Android dependency so it can be unit
// tested with a plain host g++ build (see scripts/klein_qk_norm_rope_selftest.cpp) before ever
// running on device.
//
// Math reference (flux2/model.py upstream):
//   RMSNorm:   rrms = 1 / sqrt(mean(x^2, dim=-1) + 1e-6); out = x * rrms * scale
//   apply_rope: pairs (x[2p], x[2p+1]) for p in [0, head_dim/2) are rotated by a 2x2 matrix
//               [[cos,-sin],[sin,cos]] carried in `rope` as 4 floats per pair, per token:
//               [cos, -sin, sin, cos] (this is exactly the layout jni_diffusion.cpp's on-device
//               RoPE generator already produces, validated against upstream EmbedND to 3.02e-6).
#ifndef POCKETTAVERN_KLEIN_QK_NORM_ROPE_HPP
#define POCKETTAVERN_KLEIN_QK_NORM_ROPE_HPP

#include <cmath>
#include <cstddef>

namespace pockettavern {

// Applies RMSNorm (per head, per token, over the head_dim axis) followed by RoPE, in place, to
// one of Q or K.
//
// Layout: `qk` is [heads, tokens, head_dim] flattened head-major (matches the NPU raw-qkv
// artifact's output layout and every other Klein engine's internal convention).
// `scale` is the learned RMSNorm scale, length head_dim, shared across all heads/tokens.
// `rope` is [tokens, (head_dim/2)*4], the same buffer already used for the fused-artifact's pe
// input (stride kPeStride=256 for head_dim=128).
inline void ApplyQkNormRope(float* qk, int heads, int tokens, int head_dim,
                             const float* scale, const float* rope) {
  constexpr float kEps = 1e-6f;
  const int pairs = head_dim / 2;
  for (int h = 0; h < heads; ++h) {
    for (int t = 0; t < tokens; ++t) {
      float* vec = qk + (static_cast<size_t>(h) * tokens + t) * head_dim;
      double sum_sq = 0.0;
      for (int d = 0; d < head_dim; ++d) sum_sq += static_cast<double>(vec[d]) * vec[d];
      const float rrms = static_cast<float>(1.0 / std::sqrt(sum_sq / head_dim + kEps));
      for (int d = 0; d < head_dim; ++d) vec[d] = vec[d] * rrms * scale[d];

      const float* rope_token = rope + static_cast<size_t>(t) * pairs * 4;
      for (int p = 0; p < pairs; ++p) {
        const float cos_v = rope_token[p * 4 + 0];
        const float neg_sin_v = rope_token[p * 4 + 1];
        const float sin_v = rope_token[p * 4 + 2];
        const float cos_v2 = rope_token[p * 4 + 3];
        const float x0 = vec[p * 2 + 0];
        const float x1 = vec[p * 2 + 1];
        vec[p * 2 + 0] = cos_v * x0 + neg_sin_v * x1;
        vec[p * 2 + 1] = sin_v * x0 + cos_v2 * x1;
      }
    }
  }
}

}  // namespace pockettavern

#endif  // POCKETTAVERN_KLEIN_QK_NORM_ROPE_HPP
