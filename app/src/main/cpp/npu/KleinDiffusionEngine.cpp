#include "npu/KleinDiffusionEngine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "npu/KleinComponentEngine.hpp"
#include "npu/KleinTransformerEngine.hpp"

namespace pockettavern {
namespace {

constexpr size_t kHiddenDim = 3072;   // H
constexpr size_t kImageTokens = 4096; // I (64x64 latent grid)
constexpr size_t kTextTokens = 512;   // T
constexpr size_t kContextDim = 3 * QwenTextEncoderEngine::kHiddenPerLayer; // C = 7680
constexpr size_t kPeDim = 256;        // P
constexpr size_t kLatentChannels = 128;
constexpr size_t kLatentGrid = 64;    // sqrt(kImageTokens)
constexpr size_t kOutputSize = 1024;

// Match sampling.compute_empirical_mu(image_seq_len=kImageTokens, num_steps=4) -- the
// interpolated shift used by Klein's real 4-step distilled schedule (not the asymptotic
// 200-step value). Mirrors nativeRunKleinOneStepReference's schedule exactly.
struct StepSchedule {
  std::vector<float> time_embedding;  // [256]
  double shifted_t;
  double shifted_next;
};

StepSchedule BuildStepSchedule(int step_index) {
  const double raw_t = 1.0 - 0.25 * step_index;
  const double m200 = 0.00016927 * kImageTokens + 0.45666666;
  const double m10 = 8.73809524e-05 * kImageTokens + 1.89833333;
  const double mu = m200 + (4.0 - 200.0) * (m200 - m10) / 190.0;
  const double e = std::exp(mu);
  const double shifted_t = e / (e + (1.0 / raw_t - 1.0));
  const double raw_next = raw_t - 0.25;
  const double shifted_next = raw_next == 0.0 ? 0.0 : e / (e + (1.0 / raw_next - 1.0));

  std::vector<float> time(kPeDim);
  for (size_t i = 0; i < kPeDim / 2; ++i) {
    const double omega = std::exp(-std::log(10000.0) * i / (kPeDim / 2));
    time[i] = static_cast<float>(std::cos(1000.0 * shifted_t * omega));
    time[kPeDim / 2 + i] = static_cast<float>(std::sin(1000.0 * shifted_t * omega));
  }
  return {std::move(time), shifted_t, shifted_next};
}

// Four-axis RoPE positions -- position-only, so identical every denoising step. tokens/image
// mirror nativeRunKleinOneStepReference's positions() lambda exactly.
std::vector<float> BuildRopePositions(size_t tokens, bool image) {
  std::vector<float> pe(tokens * kPeDim);
  for (size_t n = 0; n < tokens; ++n) {
    const int coords[4] = {0, image ? static_cast<int>(n / kLatentGrid) : 0,
                            image ? static_cast<int>(n % kLatentGrid) : 0,
                            image ? 0 : static_cast<int>(n)};
    size_t o = n * kPeDim;
    for (int axis = 0; axis < 4; ++axis) {
      for (int j = 0; j < 16; ++j) {
        const double a = coords[axis] / std::pow(2000.0, 2.0 * j / 32.0);
        const float c = static_cast<float>(std::cos(a)), s = static_cast<float>(std::sin(a));
        pe[o++] = c;
        pe[o++] = -s;
        pe[o++] = s;
        pe[o++] = c;
      }
    }
  }
  return pe;
}

bool WriteRgbPng(const std::string& path, const std::vector<float>& nchw) {
  if (nchw.size() != 3 * kOutputSize * kOutputSize) return false;
  std::vector<uint8_t> rgb(3 * kOutputSize * kOutputSize);
  for (size_t y = 0; y < kOutputSize; ++y) {
    for (size_t x = 0; x < kOutputSize; ++x) {
      const size_t pixel = y * kOutputSize + x;
      for (size_t c = 0; c < 3; ++c) {
        const float v = nchw[c * kOutputSize * kOutputSize + pixel];
        rgb[pixel * 3 + c] = static_cast<uint8_t>(std::clamp((v + 1.f) * 127.5f, 0.f, 255.f));
      }
    }
  }
  return stbi_write_png(path.c_str(), static_cast<int>(kOutputSize), static_cast<int>(kOutputSize),
                         3, rgb.data(), static_cast<int>(kOutputSize * 3)) != 0;
}

}  // namespace

bool KleinDiffusionEngine::Load(const std::string& npu_model_dir,
                                 const std::string& dispatch_lib_dir,
                                 const std::string& qwen_config_path,
                                 const std::string& mmap_cache_dir) {
  npu_model_dir_ = npu_model_dir;
  dispatch_lib_dir_ = dispatch_lib_dir;
  return encoder_.Load(qwen_config_path, mmap_cache_dir);
}

bool KleinDiffusionEngine::Generate(const std::string& prompt, uint32_t seed,
                                     const std::string& output_png_path,
                                     const std::function<void(int)>& progress_callback) {
  if (!encoder_.IsLoaded()) return false;
  const auto tick = [&](int phase, int total) {
    if (progress_callback) progress_callback((phase * 100) / total);
  };
  constexpr int kTotalPhases = 1 /*encode*/ + 4 /*steps*/ + 8 /*vae stages*/;
  int phase = 0;

  std::vector<float> context;
  if (!encoder_.Encode(prompt, &context) || context.size() != kTextTokens * kContextDim) {
    return false;
  }
  // Free the encoder's ~8GB weight mapping now rather than leaving it resident (even mmap'd and
  // in principle kernel-reclaimable) through the whole NPU denoise + VAE decode below -- confirmed
  // on-device that relying on reclaim alone loses the race against those phases' own allocations
  // and breaches the Memory Limiter ceiling. See QwenTextEncoderEngine::Unload()'s doc comment.
  encoder_.Unload();
  tick(++phase, kTotalPhases);

  std::mt19937 generator(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> latent(kImageTokens * kLatentChannels);
  for (float& value : latent) value = normal(generator);

  const std::vector<float> pe = BuildRopePositions(kImageTokens, /*image=*/true);
  const std::vector<float> pe_ctx = BuildRopePositions(kTextTokens, /*image=*/false);

  KleinComponentEngine components;
  if (!components.Load(dispatch_lib_dir_)) return false;

  for (int step_index = 0; step_index < 4; ++step_index) {
    const StepSchedule schedule = BuildStepSchedule(step_index);

    std::vector<float> vec, img, txt;
    if (!components.Run(npu_model_dir_ + "/time_in_Google_Tensor_G5.tflite", {&schedule.time_embedding},
                         {{1, static_cast<int32_t>(kPeDim)}}, {1, static_cast<int32_t>(kHiddenDim)}, &vec) ||
        !components.Run(npu_model_dir_ + "/img_in_Google_Tensor_G5.tflite", {&latent},
                         {{1, static_cast<int32_t>(kImageTokens), static_cast<int32_t>(kLatentChannels)}},
                         {1, static_cast<int32_t>(kImageTokens), static_cast<int32_t>(kHiddenDim)}, &img) ||
        !components.Run(npu_model_dir_ + "/txt_in_Google_Tensor_G5.tflite", {&context},
                         {{1, static_cast<int32_t>(kTextTokens), static_cast<int32_t>(kContextDim)}},
                         {1, static_cast<int32_t>(kTextTokens), static_cast<int32_t>(kHiddenDim)}, &txt)) {
      return false;
    }

    std::vector<std::vector<float>> mi, mt, ms;
    const std::vector<std::vector<int32_t>> six(6, {1, static_cast<int32_t>(kHiddenDim)}),
        three(3, {1, static_cast<int32_t>(kHiddenDim)});
    if (!components.RunMulti(npu_model_dir_ + "/mod_img_Google_Tensor_G5.tflite", {&vec},
                              {{1, static_cast<int32_t>(kHiddenDim)}}, six, &mi) ||
        !components.RunMulti(npu_model_dir_ + "/mod_txt_Google_Tensor_G5.tflite", {&vec},
                              {{1, static_cast<int32_t>(kHiddenDim)}}, six, &mt) ||
        !components.RunMulti(npu_model_dir_ + "/mod_single_Google_Tensor_G5.tflite", {&vec},
                              {{1, static_cast<int32_t>(kHiddenDim)}}, three, &ms)) {
      return false;
    }

    std::array<KleinDoubleModulation, 5> dm;
    std::array<KleinSingleModulation, 20> sm;
    for (auto& m : dm) {
      m.image_first = {mi[0], mi[1], mi[2]};
      m.image_second = {mi[3], mi[4], mi[5]};
      m.text_first = {mt[0], mt[1], mt[2]};
      m.text_second = {mt[3], mt[4], mt[5]};
    }
    for (auto& m : sm) m = {ms[0], ms[1], ms[2]};

    KleinTransformerEngine transformer;
    std::vector<float> prediction;
    if (!transformer.Forward(npu_model_dir_, dispatch_lib_dir_, pe, pe_ctx, dm, sm, &img, &txt,
                              /*single_worker_count=*/4) ||
        !components.Run(npu_model_dir_ + "/final_Google_Tensor_G5.tflite", {&img, &vec},
                         {{1, static_cast<int32_t>(kImageTokens), static_cast<int32_t>(kHiddenDim)},
                          {1, static_cast<int32_t>(kHiddenDim)}},
                         {1, static_cast<int32_t>(kImageTokens), static_cast<int32_t>(kLatentChannels)},
                         &prediction)) {
      return false;
    }

    const double delta = schedule.shifted_next - schedule.shifted_t;
    for (size_t i = 0; i < latent.size(); ++i) {
      latent[i] += static_cast<float>(delta * prediction[i]);
    }
    tick(++phase, kTotalPhases);
  }

  // Unpack [tokens, channels] latent into the VAE decoder's [channels, height, width] input.
  std::vector<float> current(kLatentChannels * kLatentGrid * kLatentGrid);
  for (size_t h = 0; h < kLatentGrid; ++h) {
    for (size_t w = 0; w < kLatentGrid; ++w) {
      for (size_t c = 0; c < kLatentChannels; ++c) {
        current[(c * kLatentGrid + h) * kLatentGrid + w] = latent[(h * kLatentGrid + w) * kLatentChannels + c];
      }
    }
  }
  latent.clear();
  latent.shrink_to_fit();

  struct Stage {
    const char* file;
    std::vector<int32_t> input;
    std::vector<int32_t> output;
  };
  const Stage stages[] = {
      {"vae_decoder_pre_mid_Google_Tensor_G5.tflite", {1, 128, 64, 64}, {1, 512, 128, 128}},
      {"vae_decoder_up_3_Google_Tensor_G5.tflite", {1, 512, 128, 128}, {1, 512, 256, 256}},
      {"vae_decoder_up_2_Google_Tensor_G5.tflite", {1, 512, 256, 256}, {1, 512, 512, 512}},
      {"vae_decoder_up_1_Google_Tensor_G5.tflite", {1, 512, 512, 512}, {1, 256, 1024, 1024}},
      {"vae_decoder_up_0_block_0_Google_Tensor_G5.tflite", {1, 256, 1024, 1024}, {1, 128, 1024, 1024}},
      {"vae_decoder_up_0_block_1_Google_Tensor_G5.tflite", {1, 128, 1024, 1024}, {1, 128, 1024, 1024}},
      {"vae_decoder_up_0_block_2_Google_Tensor_G5.tflite", {1, 128, 1024, 1024}, {1, 128, 1024, 1024}},
      {"vae_decoder_up_0_head_Google_Tensor_G5.tflite", {1, 128, 1024, 1024}, {1, 3, 1024, 1024}},
  };
  for (const Stage& stage : stages) {
    std::vector<float> next;
    if (!components.Run(npu_model_dir_ + "/" + stage.file, {&current}, {stage.input}, stage.output, &next)) {
      return false;
    }
    current.swap(next);
    tick(++phase, kTotalPhases);
  }

  return WriteRgbPng(output_png_path, current);
}

}  // namespace pockettavern
