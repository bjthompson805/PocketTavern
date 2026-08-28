// NpuUnetEngine: runs SDXL's UNet as 36 separately-dispatched, RESHAPE-wrapped LiteRT
// CompiledModels on the Tensor G5 NPU, in place of MNN's own UNet Module/Session. Each piece is
// created, run, and closed per forward() call (per diffusion step) -- this bounds peak dma-buf
// memory near the largest single piece rather than the sum of all 36+ dispatch ops, and is the
// exact architecture validated via NpuDiagnostic.kt this session (confirmed correct: bit-exact
// vs a single merged CompiledModel; confirmed viable under real load: a full real 20-step
// diffusion run completed cleanly where a single 73-op merged CompiledModel died at step 6 with
// a real lowmemorykiller kill at dmabuf_rss=12.7GB). Every piece file already has the
// RESHAPE-wrap fix baked in at conversion time (see build_reshape_wrapped_piece.py) -- this
// class does not need to know about that; it only sees plain external float32 in/out per piece.
//
// This class owns a real C API surface (litert/c/*.h from google-ai-edge/LiteRT, linked against
// libLiteRt.so already bundled via the app's litert:2.2.0 dependency) -- NOT the Kotlin/JNI
// wrapper (com.google.ai.edge.litert) that every other NPU experiment this session used. No JNI
// round-trip, no Kotlin involved: this is called directly from StableDiffusionXL::unet() inside
// the existing native diffusion loop, alongside the (unchanged) MNN text-encoder/scheduler/VAE
// modules.
//
// No exceptions: the vendored MNN build compiles with -fno-exceptions, and a C++ exception
// escaping across the JNI boundary back to Java is undefined behavior regardless of which target
// this file itself is compiled with. Every failure path logs via MNN_ERROR (matching
// stable_diffusion_xl.cpp's own convention) and returns false/empty, exactly like the rest of
// this native codebase already does -- no throw, no try/catch anywhere in this class.
//
// Two-phase create-then-load, mirroring MnnDiffusionEngine's own nativeCreate/nativeLoad split:
// the constructor does no I/O and cannot fail; Load() does the real work (LiteRT environment
// setup, piece file existence checks) and reports success/failure explicitly.
//
// Thread-unsafe by design, matching its one caller: StableDiffusionXL::unet() already runs on a
// dedicated background thread (see MnnDiffusionEngine's IO-dispatcher wrapping of the blocking
// native call), so Load()/forward() are plain blocking calls, no internal locking or async API.

#ifndef POCKETTAVERN_NPU_UNET_ENGINE_HPP
#define POCKETTAVERN_NPU_UNET_ENGINE_HPP

#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"

namespace pockettavern {

// One external UNet input tensor. Order matches StableDiffusionXL::unet()'s existing MNN call
// site (sample/timestep/encoder_hidden_states/text_embeds/time_ids) -- see
// stable_diffusion_xl.cpp's mModules[2]->onForward(...) call this replaces. Shapes below are
// per-row; every vector's actual size is batch * (per-row element count), batch matching
// whatever was passed to Load() -- see that method's doc. For batch=2 (real CFG), sample/t_emb/
// text_embeds/time_ids/encoder_hidden_states are literally the same batch=2 tensors
// StableDiffusionXL::unet() already builds for the MNN path (uncond row then cond row); nothing
// needs to be duplicated or sliced by hand the way the batch=1 conditional-only path does.
struct NpuUnetInputs {
  std::vector<float> sample;                 // [*,4,128,128]
  std::vector<float> t_emb;                  // [*,320] -- diffusers' weight-free Timesteps(t), NOT a raw scalar timestep
  std::vector<float> encoder_hidden_states;  // [*,77,2048]
  std::vector<float> text_embeds;            // [*,1280]
  std::vector<float> time_ids;               // [*,6]
};

class NpuUnetEngine {
 public:
  NpuUnetEngine();
  ~NpuUnetEngine();

  NpuUnetEngine(const NpuUnetEngine&) = delete;
  NpuUnetEngine& operator=(const NpuUnetEngine&) = delete;

  // Creates the LiteRT environment and verifies all 36 wrapped piece files exist under
  // model_dir. dispatch_lib_dir is the directory containing libLiteRtDispatch_GoogleTensor.so
  // (on Android, the app's ApplicationInfo.nativeLibraryDir -- there's no way to derive this
  // from C++ alone, must be passed down from the Kotlin/JNI caller). batch must match how
  // model_dir's 36 pieces were actually AOT-compiled (1 = conditional-only, no CFG; 2 = real
  // CFG, uncond+cond in one dispatch per piece) -- passing the wrong value doesn't fail Load()
  // itself but produces wrong-sized buffers (and either a hard LiteRT error or silently wrong
  // output) the first time forward() runs a piece. Safe to call at most once per instance
  // (matches MnnDiffusionEngine's create-then-load-once pattern) -- returns false and logs via
  // MNN_ERROR on any failure (missing files, environment creation failure), leaving the engine
  // unusable (IsLoaded() stays false; forward() will fail without touching any NPU state). Does
  // NOT run anything -- pieces are still created/closed per forward() call, not pre-warmed here.
  bool Load(std::string model_dir, const std::string& dispatch_lib_dir, int batch = 1);

  bool IsLoaded() const { return env_ != nullptr; }

  int batch() const { return batch_; }

  // Runs one real UNet forward pass (one diffusion step, batch=1 -- CFG batching, if used, is
  // the CALLER's responsibility: call forward() twice, once per conditioning, and combine the
  // two results; see project notes on why these pieces are batch=1 only). Writes the
  // [1,4,128,128] noise-prediction tensor into *out_noise_pred and returns true on success.
  // Returns false (logging the specific LiteRtStatus/piece via MNN_ERROR) on any failure --
  // *out_noise_pred is left unmodified in that case, never partially written.
  bool forward(const NpuUnetInputs& inputs, std::vector<float>* out_noise_pred);

 private:
  struct PieceSpec {
    const char* name;
    const char* file_name;
    std::vector<const char*> input_roles;
    std::vector<const char*> output_roles;
  };

  // Transcribed from full_unet_pieces.json (build_full_unet_wrapped.py), same source of truth
  // already used for NpuDiagnostic.kt's UNET_PIECES -- keep both in sync if the piece set ever
  // changes; do not hand-edit one without regenerating the other from the same manifest.
  static const std::vector<PieceSpec>& Pieces();

  // One piece's create->write inputs->run->read outputs->close cycle. Returns false (logging
  // which piece and why) on any failure. Not exposed publicly -- forward() is the only entry
  // point; a single piece has no meaning outside a full UNet pass.
  bool RunPiece(const PieceSpec& piece, const std::vector<const std::vector<float>*>& piece_inputs,
                std::vector<std::vector<float>>* out_results);

  std::string model_dir_;
  LiteRtEnvironment env_ = nullptr;
  int batch_ = 1;
};

}  // namespace pockettavern

#endif  // POCKETTAVERN_NPU_UNET_ENGINE_HPP
