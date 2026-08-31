// Host-buildable (no NDK/LiteRT needed) correctness test for
// app/src/main/cpp/npu/klein_qk_norm_rope.hpp against a PyTorch-generated reference.
//
// Build+run:
//   g++ -std=c++17 -O2 -I app/src/main/cpp/npu scripts/klein_qk_norm_rope_selftest.cpp \
//       -o /tmp/klein_qk_norm_rope_selftest
//   /tmp/klein_qk_norm_rope_selftest /tmp/klein_qk_norm_rope_ref
//
// Reference files come from scratch/klein_qk_norm_rope_reference.py in the litert-torch repo.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "klein_qk_norm_rope.hpp"

namespace {
std::vector<float> ReadBin(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
  std::vector<float> data(count);
  f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
  if (!f) { std::fprintf(stderr, "short read on %s\n", path.c_str()); std::exit(1); }
  return data;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 5) { std::fprintf(stderr, "usage: %s <ref_dir> [heads tokens head_dim]\n", argv[0]); return 2; }
  const std::string dir = argv[1];
  const int kHeads = argc == 5 ? std::atoi(argv[2]) : 24;
  const int kTokens = argc == 5 ? std::atoi(argv[3]) : 37;
  const int kHeadDim = argc == 5 ? std::atoi(argv[4]) : 128;

  auto q = ReadBin(dir + "/q_raw.bin", static_cast<size_t>(kHeads) * kTokens * kHeadDim);
  auto scale = ReadBin(dir + "/scale.bin", kHeadDim);
  auto rope = ReadBin(dir + "/rope.bin", static_cast<size_t>(kTokens) * (kHeadDim / 2) * 4);
  auto expected = ReadBin(dir + "/q_expected.bin", static_cast<size_t>(kHeads) * kTokens * kHeadDim);

  pockettavern::ApplyQkNormRope(q.data(), kHeads, kTokens, kHeadDim, scale.data(), rope.data());

  float max_abs_diff = 0.f;
  double sum_abs_diff = 0.0, sum_abs_ref = 0.0;
  for (size_t i = 0; i < q.size(); ++i) {
    const float diff = std::fabs(q[i] - expected[i]);
    if (diff > max_abs_diff) max_abs_diff = diff;
    sum_abs_diff += diff;
    sum_abs_ref += std::fabs(expected[i]);
  }
  const double mean_abs_diff = sum_abs_diff / static_cast<double>(q.size());
  const double mean_abs_ref = sum_abs_ref / static_cast<double>(q.size());
  std::printf("maxAbsDiff=%.6e meanAbsDiff=%.6e meanAbsRef=%.6e\n", max_abs_diff, mean_abs_diff, mean_abs_ref);

  const bool ok = max_abs_diff < 1e-3f;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
