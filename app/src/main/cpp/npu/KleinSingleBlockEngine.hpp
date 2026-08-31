// KleinSingleBlockEngine: runs FLUX.2 [klein]'s single_blocks.0 chunked/flash-attention design
// (docs/flux2-klein-conversion.md step 7/8) natively. It uses five AOT pieces: qkv_proj,
// flash_step_init, flash_step, attn_finalize, and out_proj. The zero-copy path chains them
// through DmaBuf (4 query chunks x 1152 = 4608 tokens) and only reads the final block result.
//
// Purpose: confirm real C++ performance and memory behavior for this design before scaling to
// the other 24 blocks or wiring into a production engine. The Kotlin/JVM diagnostic hit a real
// Dalvik heap ceiling keeping `mlp` (~85 MiB/chunk) resident and had to fall back to per-chunk
// disk I/O; that ceiling is JVM-managed-heap-specific (see docs's correction note) and does not
// apply here. The host-copy reference path keeps q/k/v/mlp/attn in native vectors; the optimized
// path retains q/k/v/mlp and attention in DmaBuf, avoiding their host round-trips entirely.
//
// Uses the real LiteRT C API and no exceptions (matching this codebase's -fno-exceptions build).
// Compiled models are persistent for the lifetime of the engine; their DmaBufs are destroyed only
// after their compiled model, as required by the Google Tensor dispatcher.
//
// Two-phase create-then-load, matching NpuUnetEngine. Thread-unsafe by design (single-threaded
// smoke test caller only).

#ifndef POCKETTAVERN_KLEIN_SINGLE_BLOCK_ENGINE_HPP
#define POCKETTAVERN_KLEIN_SINGLE_BLOCK_ENGINE_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_tensor_buffer.h"

namespace pockettavern {

class KleinSingleBlockEngine {
 public:
  KleinSingleBlockEngine();
  ~KleinSingleBlockEngine();

  KleinSingleBlockEngine(const KleinSingleBlockEngine&) = delete;
  KleinSingleBlockEngine& operator=(const KleinSingleBlockEngine&) = delete;

  // Verifies the 4 required piece files exist under model_dir and creates the LiteRT
  // environment. dispatch_lib_dir is context.applicationInfo.nativeLibraryDir (passed down from
  // Kotlin, same as NpuUnetEngine::Load). Returns false and logs via NPU_LOGE on any failure.
  // block_index selects the learned QKV/output-projection weights.  The attention artifacts are
  // shared because those kernels contain no learned block parameters.
  bool Load(std::string model_dir, const std::string& dispatch_lib_dir, int block_index = 0);

  bool IsLoaded() const { return env_ != nullptr; }

  // Releases persistent DmaBuf input/output allocations while keeping compiled model handles.
  // Used to measure and enforce a memory-bounded cache across denoising steps.
  void ReleaseCachedTensorBuffers();

  // Runs the full 24-dispatch chunked forward for single_blocks.0 at the real production shape
  // (512 text + 4096 image = 4608 tokens, chunk=1152). x: [4608*3072], pe: [4608*256],
  // mod_shift/mod_scale/mod_gate: [3072] each. Writes the [4608*3072] result into *out and
  // returns true on success; *out is left unmodified on failure.
  bool forward(const std::vector<float>& x, const std::vector<float>& pe,
               const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
               const std::vector<float>& mod_gate, std::vector<float>* out);

  // Same forward pass, but retains intermediates in DmaBuf and chains compatible kernels
  // directly. Only the final block output is read back to host memory.
  bool forwardZeroCopyPooled(const std::vector<float>& x, const std::vector<float>& pe,
                             const std::vector<float>& mod_shift,
                             const std::vector<float>& mod_scale,
                             const std::vector<float>& mod_gate,
                             std::vector<float>* out);

  // Performance experiment: submit the four independent QKV/MLP chunk dispatches concurrently.
  // Each dispatch uses a separate compiled model and DmaBuf set; dependent attention stays serial.
  bool forwardZeroCopyPooledParallelQkv(const std::vector<float>& x,
                                        const std::vector<float>& pe,
                                        const std::vector<float>& mod_shift,
                                        const std::vector<float>& mod_scale,
                                        const std::vector<float>& mod_gate,
                                        std::vector<float>* out);

  // Performance experiment: additionally submit the four independent attention-query chains
  // concurrently. Each chain has isolated compiled models and DmaBuf state.
  bool forwardZeroCopyPooledParallelQkvAttention(
      const std::vector<float>& x, const std::vector<float>& pe,
      const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
      const std::vector<float>& mod_gate, std::vector<float>* out);

  // Performance experiment: also submit the four independent output-projection chunks in
  // parallel, using isolated compiled models and output buffers.
  bool forwardZeroCopyPooledFullyParallel(
      const std::vector<float>& x, const std::vector<float>& pe,
      const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
      const std::vector<float>& mod_gate, std::vector<float>* out);

  // Runs the fully parallel topology in waves of `worker_count` chunks (1..4), for Tensor G5
  // concurrency tuning. The same numerical graph is used for every worker count.
  bool forwardZeroCopyPooledWithWorkers(
      const std::vector<float>& x, const std::vector<float>& pe,
      const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
      const std::vector<float>& mod_gate, std::vector<float>* out, int worker_count);

  // Compatibility probe for a production inter-block handoff: feeds each final `_out` DmaBuf
  // directly into QKV's activation input, without copying through host memory.
  bool forwardZeroCopyPooledInterBlockProbe(
      const std::vector<float>& x, const std::vector<float>& pe,
      const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
      const std::vector<float>& mod_gate, std::vector<float>* out);

 private:
  struct CachedPiece;
  // Host-copy reference path. Compiled models are retained; tensor buffers are local to a call.
  bool RunPiece(const std::string& file_name,
                const std::vector<const std::vector<float>*>& inputs,
                const std::vector<std::vector<int32_t>>& input_shapes,
                const std::vector<std::vector<int32_t>>& output_shapes,
                std::vector<std::vector<float>>* out_results);
  bool RunCachedDirect(const std::string& cache_key,
                       const std::vector<LiteRtTensorBuffer>& input_buffers,
                       const std::vector<LiteRtTensorBuffer>* output_buffers = nullptr);
  CachedPiece* PrepareCachedPiece(const std::string& file_name,
                                  const std::vector<std::vector<int32_t>>& input_shapes,
                                  const std::vector<std::vector<int32_t>>& output_shapes,
                                  const std::string& cache_key);
  bool forwardZeroCopyPooledImpl(const std::vector<float>& x, const std::vector<float>& pe,
                                 const std::vector<float>& mod_shift,
                                 const std::vector<float>& mod_scale,
                                 const std::vector<float>& mod_gate, std::vector<float>* out,
                                 bool parallel_qkv, bool parallel_attention,
                                 bool parallel_output, bool probe_interblock, int worker_count);

  std::string model_dir_;
  std::string qkv_file_;
  std::string out_file_;
  LiteRtEnvironment env_ = nullptr;
  std::map<std::string, std::unique_ptr<CachedPiece>> cached_pieces_;
};

}  // namespace pockettavern

#endif  // POCKETTAVERN_KLEIN_SINGLE_BLOCK_ENGINE_HPP
