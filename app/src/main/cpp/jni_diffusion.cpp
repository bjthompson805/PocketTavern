#include <jni.h>
#include <android/log.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <MNN/MNNForwardType.h>
#include "diffusion/diffusion.hpp"
#include "diffusion/stable_diffusion_xl.hpp"
#include "npu/NpuUnetEngine.hpp"
#include "npu/NpuTextEncoderEngine.hpp"
#include "npu/KleinSingleBlockEngine.hpp"
#include "npu/KleinDoubleBlockEngine.hpp"
#include "npu/KleinComponentEngine.hpp"
#include "npu/KleinTransformerEngine.hpp"
#include "npu/QwenTextEncoderEngine.hpp"
#include "npu/KleinDiffusionEngine.hpp"

#define LOG_TAG "PocketTavernDiffusion"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace MNN::DIFFUSION;

namespace {

std::string jstringToStd(JNIEnv* env, jstring s) {
    if (s == nullptr) {
        return std::string();
    }
    const char* chars = env->GetStringUTFChars(s, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(s, chars);
    return result;
}

bool ReadFloatFile(const std::string& path, size_t expectedElements, std::vector<float>* values) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamsize byteCount = file.tellg();
    if (byteCount != static_cast<std::streamsize>(expectedElements * sizeof(float))) return false;
    values->resize(expectedElements);
    file.seekg(0);
    return static_cast<bool>(file.read(reinterpret_cast<char*>(values->data()), byteCount));
}

bool WriteFloatFile(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(file);
}

bool WriteRgbPpm(const std::string& path, const std::vector<float>& nchw) {
    constexpr size_t kWidth = 1024, kHeight = 1024;
    if (nchw.size() != 3 * kWidth * kHeight) return false;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file << "P6\n" << kWidth << " " << kHeight << "\n255\n";
    for (size_t y = 0; y < kHeight; ++y) for (size_t x = 0; x < kWidth; ++x) {
        const size_t pixel = y * kWidth + x;
        const auto byte = [&](size_t channel) { return static_cast<unsigned char>(std::clamp((nchw[channel * kWidth * kHeight + pixel] + 1.f) * 127.5f, 0.f, 255.f)); };
        const unsigned char rgb[] = {byte(0), byte(1), byte(2)};
        file.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
    }
    return static_cast<bool>(file);
}

// Wraps a Kotlin (Int) -> Unit lambda (a plain kotlin.jvm.functions.Function1 object from the
// Kotlin side) so it can be handed to StableDiffusionXL::runXL()'s
// std::function<void(int)> progressCallback. Non-copyable on purpose: it owns two global refs,
// and std::function normally takes its callable by value/copy -- passing this into runXL() via
// std::ref() (see nativeGenerateXL below) instead of by value is what keeps that safe. Called
// synchronously on the same thread that entered nativeGenerateXL (runXL() has no internal
// worker thread -- it blocks until the whole generation is done), so this needs no
// AttachCurrentThread/DetachCurrentThread handling; the JNIEnv* handed to nativeGenerateXL stays
// valid for the whole call.
class KotlinProgressCallback {
public:
    KotlinProgressCallback(const KotlinProgressCallback&) = delete;
    KotlinProgressCallback& operator=(const KotlinProgressCallback&) = delete;

    KotlinProgressCallback(JNIEnv* env, jobject callback) : mEnv(env) {
        mCallback = env->NewGlobalRef(callback);
        jclass callbackClass = env->GetObjectClass(mCallback);
        mInvokeMethod = env->GetMethodID(callbackClass, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
        env->DeleteLocalRef(callbackClass);
        mIntegerClass = static_cast<jclass>(env->NewGlobalRef(env->FindClass("java/lang/Integer")));
        mIntegerCtor = env->GetMethodID(mIntegerClass, "<init>", "(I)V");
    }

    ~KotlinProgressCallback() {
        mEnv->DeleteGlobalRef(mCallback);
        mEnv->DeleteGlobalRef(mIntegerClass);
    }

    void operator()(int progress) {
        jobject boxed = mEnv->NewObject(mIntegerClass, mIntegerCtor, progress);
        jobject result = mEnv->CallObjectMethod(mCallback, mInvokeMethod, boxed);
        mEnv->DeleteLocalRef(boxed);
        if (result != nullptr) {
            mEnv->DeleteLocalRef(result);
        }
        // A plain progress-reporting lambda isn't expected to throw, but an uncaught pending
        // JNI exception here would corrupt the next JNI call runXL() makes internally (its own
        // Module::onForward calls between progress ticks) -- guard defensively.
        if (mEnv->ExceptionCheck()) {
            mEnv->ExceptionDescribe();
            mEnv->ExceptionClear();
        }
    }

private:
    JNIEnv* mEnv;
    jobject mCallback;
    jclass mIntegerClass;
    jmethodID mInvokeMethod;
    jmethodID mIntegerCtor;
};

}  // namespace

extern "C" {

// modelType matches MNN::DIFFUSION::DiffusionModelType (0 = SD1.5, 4 = SDXL, etc. -- see
// diffusion.hpp). Backend is always MNN_FORWARD_CPU: not exposed as a parameter since this
// build has MNN_OPENCL compiled out (confirmed dead path for this project, see
// mnn_sdxl_android_pipeline memory) and there is currently no other real choice.
JNIEXPORT jlong JNICALL
Java_com_pockettavern_app_data_local_inference_MnnDiffusionBridge_nativeCreate(
        JNIEnv* env, jobject /*thiz*/, jstring modelPath, jint modelType, jint memoryMode,
        jstring npuUnetModelPath, jstring dispatchLibDir) {
    std::string path = jstringToStd(env, modelPath);
    std::string npuPath = jstringToStd(env, npuUnetModelPath);
    std::string dispatchPath = jstringToStd(env, dispatchLibDir);
    auto* diffusion = Diffusion::createDiffusion(
            path, static_cast<DiffusionModelType>(modelType), MNN_FORWARD_CPU, memoryMode);
    if (diffusion == nullptr) {
        LOGE("createDiffusion returned null for modelType=%d path=%s", modelType, path.c_str());
        return 0;
    }
    if (!npuPath.empty()) {
        if (modelType != STABLE_DIFFUSION_XL) {
            LOGE("NPU was requested for unsupported diffusion modelType=%d", modelType);
            delete diffusion;
            return 0;
        }
        auto* sdxl = static_cast<StableDiffusionXL*>(diffusion);

        std::string unetDir = npuPath;
        std::string teDir = npuPath;
        struct stat st{};
        if (::stat((npuPath + "/unet").c_str(), &st) == 0) {
            unetDir = npuPath + "/unet";
        }
        if (::stat((npuPath + "/text_encoder").c_str(), &st) == 0) {
            teDir = npuPath + "/text_encoder";
        }

        // Configure NPU UNet if UNet piece files exist
        if (::stat((unetDir + "/unet_piece_00_embed.tflite").c_str(), &st) == 0 ||
            ::stat((unetDir + "/unet_piece_00_embed_b2.tflite").c_str(), &st) == 0 ||
            ::stat((npuPath + "/unet_piece_00_embed.tflite").c_str(), &st) == 0) {
            if (!sdxl->configureNpuUnet(unetDir, dispatchPath)) {
                LOGE("Failed to configure NPU UNet");
                delete diffusion;
                return 0;
            }
            LOGE("Configured LiteRT NPU UNet for SDXL from %s", unetDir.c_str());
        }

        // Configure NPU Text Encoder if Text Encoder piece files exist
        if (::stat((teDir + "/text_encoder_b2_wrapped.tflite").c_str(), &st) == 0 ||
            ::stat((npuPath + "/text_encoder_b2_wrapped.tflite").c_str(), &st) == 0) {
            if (!sdxl->configureNpuTextEncoder(teDir, dispatchPath)) {
                LOGE("Failed to configure NPU Text Encoder");
                delete diffusion;
                return 0;
            }
            LOGE("Configured LiteRT NPU Text Encoder for SDXL from %s", teDir.c_str());
        }
    }
    return reinterpret_cast<jlong>(diffusion);
}

JNIEXPORT jboolean JNICALL
Java_com_pockettavern_app_data_local_inference_MnnDiffusionBridge_nativeLoad(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    if (handle == 0) {
        return JNI_FALSE;
    }
    auto* diffusion = reinterpret_cast<Diffusion*>(handle);
    return diffusion->load() ? JNI_TRUE : JNI_FALSE;
}

// SDXL-specific: calls StableDiffusionXL::runXL() directly, not the base class's simplified
// run() (which hardcodes cfgScale=5.0 and an empty negative prompt) -- runXL() exposes a real
// negative prompt and cfgScale instead of those hardcoded values. Only valid when nativeCreate's
// modelType was STABLE_DIFFUSION_XL (4) -- the
// static_cast below is unchecked, matching this JNI layer being the one place that always knows
// the real dynamic type it asked createDiffusion() for (see CMakeLists.txt's comment on why
// static_cast, not dynamic_cast/RTTI, is used here).
JNIEXPORT jboolean JNICALL
Java_com_pockettavern_app_data_local_inference_MnnDiffusionBridge_nativeGenerateXL(
        JNIEnv* env, jobject /*thiz*/, jlong handle,
        jstring prompt, jstring negativePrompt, jstring outputPath,
        jint width, jint height, jint steps, jint seed, jfloat cfgScale,
        jobject progressCallback) {
    if (handle == 0) {
        return JNI_FALSE;
    }
    auto* sdxl = static_cast<StableDiffusionXL*>(reinterpret_cast<Diffusion*>(handle));

    std::string promptStd = jstringToStd(env, prompt);
    std::string negativePromptStd = jstringToStd(env, negativePrompt);
    std::string outputPathStd = jstringToStd(env, outputPath);

    KotlinProgressCallback callback(env, progressCallback);
    bool ok = sdxl->runXL(promptStd, negativePromptStd, outputPathStd, width, height, steps, seed,
                           cfgScale, std::ref(callback));
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_pockettavern_app_data_local_inference_MnnDiffusionBridge_nativeDestroy(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    if (handle == 0) {
        return;
    }
    // Diffusion's destructor is virtual -- deleting through the base pointer correctly runs
    // ~StableDiffusionXL() (or whichever concrete type), no downcast needed for this call.
    delete reinterpret_cast<Diffusion*>(handle);
}

// THROWAWAY smoke test for NpuUnetEngine (see npu/NpuUnetEngine.hpp) -- confirms the native
// LiteRT C API integration actually works on-device (Load() + one real forward() call with
// synthetic sin() input, same generation formula NpuDiagnostic.kt's runFullUnetSeparate used in
// Kotlin all session) before wiring it into StableDiffusionXL::unet()'s real call site. Not part
// of MnnDiffusionBridge on purpose -- this has nothing to do with the MNN pipeline; delete once
// the real unet() integration lands and this is superseded.
//
// modelDir: <filesDir>/unet_wrapped (the 36 wrapped piece files, already pushed for
// NpuDiagnostic.kt's own testing this session).
// dispatchLibDir: context.applicationInfo.nativeLibraryDir (where
// libLiteRtDispatch_GoogleTensor.so lives on-device) -- C++ has no way to derive this itself,
// must come from Kotlin.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunUnetEngineSmoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir) {
    std::string modelDirStd = jstringToStd(env, modelDir);
    std::string dispatchLibDirStd = jstringToStd(env, dispatchLibDir);

    pockettavern::NpuUnetEngine engine;
    if (!engine.Load(modelDirStd, dispatchLibDirStd)) {
        return env->NewStringUTF("FAILED: Load() returned false, see logcat tag PocketTavernDiffusion");
    }

    pockettavern::NpuUnetInputs inputs;
    auto fill = [](std::vector<float>* v, size_t n) {
        v->resize(n);
        for (size_t i = 0; i < n; ++i) {
            (*v)[i] = static_cast<float>(std::sin(static_cast<double>(i)) * 0.1);
        }
    };
    fill(&inputs.sample, 1 * 4 * 128 * 128);
    fill(&inputs.t_emb, 1 * 320);
    fill(&inputs.text_embeds, 1 * 1280);
    fill(&inputs.time_ids, 1 * 6);
    fill(&inputs.encoder_hidden_states, 1 * 77 * 2048);

    std::vector<float> out;
    auto start = std::chrono::steady_clock::now();
    bool ok = engine.forward(inputs, &out);
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    if (!ok) {
        return env->NewStringUTF("FAILED: forward() returned false, see logcat tag PocketTavernDiffusion");
    }

    bool hasNanOrInf = false;
    float maxAbs = 0.0f;
    for (float v : out) {
        if (std::isnan(v) || std::isinf(v)) hasNanOrInf = true;
        float a = std::fabs(v);
        if (a > maxAbs) maxAbs = a;
    }

    char buf[512];
    snprintf(buf, sizeof(buf),
             "OK elapsedMs=%lld outputSize=%zu hasNanOrInf=%d maxAbs=%f out[0..4]=%f,%f,%f,%f,%f",
             static_cast<long long>(elapsedMs), out.size(), hasNanOrInf, maxAbs,
             out.size() > 0 ? out[0] : 0.f, out.size() > 1 ? out[1] : 0.f,
             out.size() > 2 ? out[2] : 0.f, out.size() > 3 ? out[3] : 0.f,
             out.size() > 4 ? out[4] : 0.f);
    LOGE("NpuUnetEngine smoke test: %s", buf);
    return env->NewStringUTF(buf);
}

// THROWAWAY correctness check for batch=2 (real CFG) pieces -- confirms the RESHAPE-wrap fix
// (proven at batch=1 via a real 20-step diffusion loop, see docs/npu-unet-conversion.md) still
// holds at batch=2 before wiring npuBatch=2 into any real generation path. Feeds the SAME
// per-row synthetic data into both batch rows (sample/t_emb duplicated verbatim; text_embeds/
// encoder_hidden_states/time_ids duplicated too, i.e. a degenerate "uncond==cond" case) and
// checks two things: (1) row 0 and row 1 of the output are bit-identical -- since nothing in a
// standard UNet2D forward pass mixes across the batch dimension (GroupNorm/LayerNorm normalize
// per-sample, not across batch), identical inputs per row MUST produce identical outputs per
// row; any divergence means the RESHAPE-wrap fix (or something else about batch=2 execution)
// broke that isolation. (2) row 0 matches the already-validated batch=1 smoke test's known
// output for this same sin()-generated input (see nativeRunUnetEngineSmoke above) to a tight
// tolerance -- confirms batch=2 doesn't silently change the actual computed values.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunUnetEngineBatch2Smoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir) {
    std::string modelDirStd = jstringToStd(env, modelDir);
    std::string dispatchLibDirStd = jstringToStd(env, dispatchLibDir);

    pockettavern::NpuUnetEngine engine;
    if (!engine.Load(modelDirStd, dispatchLibDirStd, /*batch=*/2)) {
        return env->NewStringUTF("FAILED: Load(batch=2) returned false, see logcat tag PocketTavernDiffusion");
    }

    auto fillRow = [](std::vector<float>* v, size_t rowElements) {
        v->resize(2 * rowElements);
        for (size_t i = 0; i < rowElements; ++i) {
            float value = static_cast<float>(std::sin(static_cast<double>(i)) * 0.1);
            (*v)[i] = value;
            (*v)[rowElements + i] = value;  // row 1 == row 0
        }
    };

    pockettavern::NpuUnetInputs inputs;
    fillRow(&inputs.sample, 4 * 128 * 128);
    fillRow(&inputs.t_emb, 320);
    fillRow(&inputs.text_embeds, 1280);
    fillRow(&inputs.time_ids, 6);
    fillRow(&inputs.encoder_hidden_states, 77 * 2048);

    // Run more than one forward on the same engine/environment. The production diffusion loop
    // does this once per denoising step; repeating here catches teardown leaks or stale-dispatch
    // state that a one-shot smoke test cannot.
    constexpr int kSmokePasses = 3;
    std::vector<float> out;
    std::vector<long long> passMs;
    passMs.reserve(kSmokePasses);
    bool ok = true;
    for (int pass = 0; pass < kSmokePasses; ++pass) {
        const auto passStart = std::chrono::steady_clock::now();
        if (!engine.forward(inputs, &out)) {
            ok = false;
            break;
        }
        passMs.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - passStart)
                             .count());
    }
    if (!ok) {
        return env->NewStringUTF("FAILED: forward(batch=2) returned false, see logcat tag PocketTavernDiffusion");
    }

    const size_t rowCount = 4 * 128 * 128;
    if (out.size() != 2 * rowCount) {
        char buf[256];
        snprintf(buf, sizeof(buf), "FAILED: unexpected output size %zu (expected %zu)", out.size(), 2 * rowCount);
        return env->NewStringUTF(buf);
    }

    bool hasNanOrInf = false;
    float maxAbs = 0.0f;
    float maxRowDiff = 0.0f;
    for (size_t i = 0; i < rowCount; ++i) {
        float a = out[i];
        float b = out[rowCount + i];
        if (std::isnan(a) || std::isinf(a) || std::isnan(b) || std::isinf(b)) hasNanOrInf = true;
        maxAbs = std::max({maxAbs, std::fabs(a), std::fabs(b)});
        maxRowDiff = std::max(maxRowDiff, std::fabs(a - b));
    }

    char buf[512];
    snprintf(buf, sizeof(buf),
             "OK passes=%zu elapsedMs=%lld,%lld,%lld outputSize=%zu hasNanOrInf=%d maxAbs=%f maxRowDiff=%.9f row0[0..4]=%f,%f,%f,%f,%f",
             passMs.size(), passMs.size() > 0 ? passMs[0] : -1, passMs.size() > 1 ? passMs[1] : -1,
             passMs.size() > 2 ? passMs[2] : -1, out.size(), hasNanOrInf, maxAbs, maxRowDiff,
             out[0], out[1], out[2], out[3], out[4]);
    LOGE("NpuUnetEngine batch=2 smoke test: %s", buf);
    return env->NewStringUTF(buf);
}

// THROWAWAY performance experiment for CFG: batch=2 AOT kernels are substantially slower than
// batch=1 on the Tensor G5. This runs two *independent* batch=1 engines concurrently to discover
// whether the NPU service can overlap the unconditional and conditional forwards. The engines do
// not share an Environment or mutable state, so each forward remains thread-safe by construction.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunTwoBatch1EnginesInParallel(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir) {
    const std::string modelDirStd = jstringToStd(env, modelDir);
    const std::string dispatchLibDirStd = jstringToStd(env, dispatchLibDir);

    pockettavern::NpuUnetEngine first;
    pockettavern::NpuUnetEngine second;
    if (!first.Load(modelDirStd, dispatchLibDirStd, /*batch=*/1) ||
        !second.Load(modelDirStd, dispatchLibDirStd, /*batch=*/1)) {
        return env->NewStringUTF("FAILED: batch-1 engine Load() returned false, see logcat tag PocketTavernDiffusion");
    }

    auto makeInputs = []() {
        pockettavern::NpuUnetInputs inputs;
        auto fill = [](std::vector<float>* values, size_t count) {
            values->resize(count);
            for (size_t i = 0; i < count; ++i) {
                (*values)[i] = static_cast<float>(std::sin(static_cast<double>(i)) * 0.1);
            }
        };
        fill(&inputs.sample, 4 * 128 * 128);
        fill(&inputs.t_emb, 320);
        fill(&inputs.text_embeds, 1280);
        fill(&inputs.time_ids, 6);
        fill(&inputs.encoder_hidden_states, 77 * 2048);
        return inputs;
    };
    const auto firstInputs = makeInputs();
    const auto secondInputs = makeInputs();
    std::vector<float> firstOutput;
    std::vector<float> secondOutput;
    bool firstOk = false;
    bool secondOk = false;

    constexpr int kPasses = 4;
    std::vector<long long> elapsedMs;
    elapsedMs.reserve(kPasses);
    for (int pass = 0; pass < kPasses; ++pass) {
        firstOk = false;
        secondOk = false;
        const auto start = std::chrono::steady_clock::now();
        std::thread firstThread([&] { firstOk = first.forward(firstInputs, &firstOutput); });
        std::thread secondThread([&] { secondOk = second.forward(secondInputs, &secondOutput); });
        firstThread.join();
        secondThread.join();
        elapsedMs.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
        if (!firstOk || !secondOk) break;
    }

    char buf[384];
    snprintf(buf, sizeof(buf),
             "OK passes=%zu elapsedMs=%lld,%lld,%lld,%lld firstOk=%d firstSize=%zu secondOk=%d secondSize=%zu",
             elapsedMs.size(), elapsedMs.size() > 0 ? elapsedMs[0] : -1,
             elapsedMs.size() > 1 ? elapsedMs[1] : -1, elapsedMs.size() > 2 ? elapsedMs[2] : -1,
             elapsedMs.size() > 3 ? elapsedMs[3] : -1, firstOk, firstOutput.size(), secondOk,
             secondOutput.size());
    LOGE("NpuUnetEngine parallel batch-1 smoke test: %s", buf);
    return env->NewStringUTF(buf);
}

// Debug-only real-CFG step: the desktop driver writes one unconditional and one conditional
// batch-1 input set, then combines the two outputs with CFG itself. This deliberately uses the
// same two independent NpuUnetEngine instances measured by the parallel batch-1 smoke test,
// avoiding the slow batch-2 AOT kernels while exercising the production C++ engine.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunParallelBatch1FileStep(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir, jstring inputDir,
        jstring outputFile) {
    const std::string modelDirStd = jstringToStd(env, modelDir);
    const std::string dispatchDirStd = jstringToStd(env, dispatchLibDir);
    const std::string inputDirStd = jstringToStd(env, inputDir);
    const std::string outputFileStd = jstringToStd(env, outputFile);

    constexpr size_t kSample = 4 * 128 * 128;
    constexpr size_t kTime = 320;
    constexpr size_t kText = 1280;
    constexpr size_t kTimeIds = 6;
    constexpr size_t kEncoder = 77 * 2048;
    auto readInputs = [&](const char* prefix, pockettavern::NpuUnetInputs* inputs) {
        const std::string base = inputDirStd + "/" + prefix;
        return ReadFloatFile(base + "sample.bin", kSample, &inputs->sample) &&
               ReadFloatFile(base + "t_emb.bin", kTime, &inputs->t_emb) &&
               ReadFloatFile(base + "text_embeds.bin", kText, &inputs->text_embeds) &&
               ReadFloatFile(base + "time_ids.bin", kTimeIds, &inputs->time_ids) &&
               ReadFloatFile(base + "ehs.bin", kEncoder, &inputs->encoder_hidden_states);
    };

    pockettavern::NpuUnetInputs uncondInputs;
    pockettavern::NpuUnetInputs condInputs;
    if (!readInputs("uncond_", &uncondInputs) || !readInputs("cond_", &condInputs)) {
        return env->NewStringUTF("FAILED: invalid parallel batch-1 input files");
    }

    pockettavern::NpuUnetEngine uncondEngine;
    pockettavern::NpuUnetEngine condEngine;
    if (!uncondEngine.Load(modelDirStd, dispatchDirStd, /*batch=*/1) ||
        !condEngine.Load(modelDirStd, dispatchDirStd, /*batch=*/1)) {
        return env->NewStringUTF("FAILED: batch-1 engine Load() returned false; see PocketTavernDiffusion");
    }

    std::vector<float> uncondOutput;
    std::vector<float> condOutput;
    bool uncondOk = false;
    bool condOk = false;
    const auto start = std::chrono::steady_clock::now();
    std::thread uncondThread([&] { uncondOk = uncondEngine.forward(uncondInputs, &uncondOutput); });
    std::thread condThread([&] { condOk = condEngine.forward(condInputs, &condOutput); });
    uncondThread.join();
    condThread.join();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    if (!uncondOk || !condOk || uncondOutput.size() != kSample || condOutput.size() != kSample) {
        return env->NewStringUTF("FAILED: parallel batch-1 NPU forward failed; see PocketTavernDiffusion");
    }
    uncondOutput.insert(uncondOutput.end(), condOutput.begin(), condOutput.end());
    if (!WriteFloatFile(outputFileStd, uncondOutput)) {
        return env->NewStringUTF("FAILED: could not write parallel batch-1 output file");
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "STEP_DONE %lldms outputSize=%zu", static_cast<long long>(elapsedMs),
             uncondOutput.size());
    LOGE("NpuUnetEngine parallel batch-1 file step: %s", buf);
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunTextEncoderEngineSmoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir) {
    std::string modelDirStd = jstringToStd(env, modelDir);
    std::string dispatchLibDirStd = jstringToStd(env, dispatchLibDir);

    pockettavern::NpuTextEncoderEngine engine;
    if (!engine.Load(modelDirStd, dispatchLibDirStd)) {
        return env->NewStringUTF("FAILED: Load() returned false, see logcat tag PocketTavernDiffusion");
    }

    // Synthetic tokens: 2 rows of 77 tokens (bos=49406, eos=49407)
    std::vector<int32_t> ids1(2 * 77, 0);
    std::vector<int32_t> ids2(2 * 77, 0);
    ids1[0] = 49406; ids1[76] = 49407;
    ids1[77] = 49406; ids1[153] = 49407;
    ids2[0] = 49406; ids2[76] = 49407;
    ids2[77] = 49406; ids2[153] = 49407;

    std::vector<float> encoderHiddenStates;
    std::vector<float> textEmbeds;
    const auto start = std::chrono::steady_clock::now();
    bool ok = engine.encode(ids1, ids2, &encoderHiddenStates, &textEmbeds);
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    if (!ok) {
        return env->NewStringUTF("FAILED: encode() returned false, see logcat tag PocketTavernDiffusion");
    }

    bool hasNanOrInf = false;
    for (float v : encoderHiddenStates) {
        if (std::isnan(v) || std::isinf(v)) hasNanOrInf = true;
    }
    for (float v : textEmbeds) {
        if (std::isnan(v) || std::isinf(v)) hasNanOrInf = true;
    }

    char buf[512];
    snprintf(buf, sizeof(buf),
             "OK elapsedMs=%lld hiddenSize=%zu embedsSize=%zu hasNanOrInf=%d hidden[0..3]=%f,%f,%f,%f embeds[0..3]=%f,%f,%f,%f",
             static_cast<long long>(elapsedMs), encoderHiddenStates.size(), textEmbeds.size(), hasNanOrInf,
             encoderHiddenStates.size() > 0 ? encoderHiddenStates[0] : 0.f,
             encoderHiddenStates.size() > 1 ? encoderHiddenStates[1] : 0.f,
             encoderHiddenStates.size() > 2 ? encoderHiddenStates[2] : 0.f,
             encoderHiddenStates.size() > 3 ? encoderHiddenStates[3] : 0.f,
             textEmbeds.size() > 0 ? textEmbeds[0] : 0.f,
             textEmbeds.size() > 1 ? textEmbeds[1] : 0.f,
             textEmbeds.size() > 2 ? textEmbeds[2] : 0.f,
             textEmbeds.size() > 3 ? textEmbeds[3] : 0.f);
    LOGE("NpuTextEncoderEngine smoke test: %s", buf);
    return env->NewStringUTF(buf);
}

// THROWAWAY perf/correctness confirmation for KleinSingleBlockEngine (see its .hpp doc comment)
// -- runs the real chunked/flash-attention design natively (no JVM heap involved at all, unlike
// NpuDiagnostic.kt's runKleinChunkedBlockProbe, which had to keep `mlp` on disk to fit the
// Dalvik-managed-heap ceiling) and diffs the result against the real PyTorch reference
// (torch_out.bin, the same file that Kotlin diagnostic already used). modelDir must contain the
// 5 compiled pieces (qkv_proj/flash_step/flash_step_init/attn_finalize/out_proj); refDir must contain
// x.bin/pe.bin/mod_shift.bin/mod_scale.bin/mod_gate.bin/torch_out.bin (both can be the same
// directory -- klein_single0/ already has both from the Kotlin diagnostic's own setup).
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinSingleBlockSmoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir, jstring refDir) {
    std::string modelDirStd = jstringToStd(env, modelDir);
    std::string dispatchLibDirStd = jstringToStd(env, dispatchLibDir);
    std::string refDirStd = jstringToStd(env, refDir);
    const std::string base = refDirStd + "/";

    constexpr size_t kTokenCount = 4608;
    constexpr size_t kHiddenSize = 3072;
    constexpr size_t kPeStride = 256;

    std::vector<float> x, pe, modShift, modScale, modGate, refOut;
    bool readOk =
        ReadFloatFile(base + "x.bin", kTokenCount * kHiddenSize, &x) &&
        ReadFloatFile(base + "pe.bin", kTokenCount * kPeStride, &pe) &&
        ReadFloatFile(base + "mod_shift.bin", kHiddenSize, &modShift) &&
        ReadFloatFile(base + "mod_scale.bin", kHiddenSize, &modScale) &&
        ReadFloatFile(base + "mod_gate.bin", kHiddenSize, &modGate) &&
        ReadFloatFile(base + "torch_out.bin", kTokenCount * kHiddenSize, &refOut);
    if (!readOk) {
        return env->NewStringUTF("FAILED: could not read reference input/output files from refDir");
    }

    pockettavern::KleinSingleBlockEngine engine;
    if (!engine.Load(modelDirStd, dispatchLibDirStd)) {
        return env->NewStringUTF("FAILED: Load() returned false, see logcat tag PocketTavernDiffusion");
    }

    std::vector<float> out;
    const auto cold_start = std::chrono::steady_clock::now();
    bool ok = engine.forwardZeroCopyPooledWithWorkers(
        x, pe, modShift, modScale, modGate, &out, /*worker_count=*/4);
    const auto cold_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - cold_start)
                               .count();
    if (!ok) {
        return env->NewStringUTF("FAILED: forwardZeroCopyPooledWithWorkers() returned false, see logcat tag PocketTavernDiffusion");
    }
    const auto warm_start = std::chrono::steady_clock::now();
    ok = engine.forwardZeroCopyPooledWithWorkers(
        x, pe, modShift, modScale, modGate, &out, /*worker_count=*/4);
    const auto warm_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - warm_start)
                               .count();
    if (!ok) {
        return env->NewStringUTF("FAILED: warm forwardZeroCopyPooledWithWorkers() returned false, see logcat tag PocketTavernDiffusion");
    }
    engine.ReleaseCachedTensorBuffers();
    const auto bounded_start = std::chrono::steady_clock::now();
    ok = engine.forwardZeroCopyPooledWithWorkers(
        x, pe, modShift, modScale, modGate, &out, /*worker_count=*/4);
    const auto bounded_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - bounded_start)
                               .count();
    if (!ok) {
        return env->NewStringUTF("FAILED: model-only-cache forward returned false, see logcat tag PocketTavernDiffusion");
    }
    if (out.size() != refOut.size()) {
        char buf[256];
        snprintf(buf, sizeof(buf), "FAILED: output size mismatch out=%zu ref=%zu", out.size(), refOut.size());
        return env->NewStringUTF(buf);
    }

    bool hasNanOrInf = false;
    float maxAbsDiff = 0.f;
    double sumAbsDiff = 0.0;
    double sumAbsRef = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        float v = out[i];
        if (std::isnan(v) || std::isinf(v)) hasNanOrInf = true;
        float diff = std::fabs(v - refOut[i]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
        sumAbsDiff += diff;
        sumAbsRef += std::fabs(refOut[i]);
    }
    double meanAbsDiff = sumAbsDiff / static_cast<double>(out.size());
    double meanAbsRef = sumAbsRef / static_cast<double>(out.size());

    char buf[512];
    snprintf(buf, sizeof(buf),
             "OK coldElapsedMs=%lld warmElapsedMs=%lld modelOnlyElapsedMs=%lld hasNanOrInf=%d maxAbsDiff=%f meanAbsDiff=%f meanAbsRef=%f out[0..4]=%f,%f,%f,%f,%f ref[0..4]=%f,%f,%f,%f,%f",
             static_cast<long long>(cold_elapsed_ms), static_cast<long long>(warm_elapsed_ms),
             static_cast<long long>(bounded_elapsed_ms), hasNanOrInf, maxAbsDiff, meanAbsDiff, meanAbsRef,
             out[0], out[1], out[2], out[3], out[4],
             refOut[0], refOut[1], refOut[2], refOut[3], refOut[4]);
    LOGE("KleinSingleBlockEngine smoke test: %s", buf);
    return env->NewStringUTF(buf);
}

// Native counterpart to runKleinDoubleChunkedBlockProbe. It measures the same validated 35
// dispatch asymmetric double-stream chain without Kotlin's managed-heap or TensorBuffer wrapper.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinDoubleBlockSmoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir, jstring refDir) {
    const std::string model_dir = jstringToStd(env, modelDir);
    const std::string dispatch_dir = jstringToStd(env, dispatchLibDir);
    const std::string base = jstringToStd(env, refDir) + "/";
    constexpr size_t kImg = 4096 * 3072;
    constexpr size_t kTxt = 512 * 3072;
    constexpr size_t kImgPe = 4096 * 256;
    constexpr size_t kTxtPe = 512 * 256;
    constexpr size_t kHidden = 3072;
    std::vector<float> img, txt, pe, pe_ctx, img1s, img1sc, img1g, img2s, img2sc, img2g;
    std::vector<float> txt1s, txt1sc, txt1g, txt2s, txt2sc, txt2g, ref_img, ref_txt;
    auto read = [&](const char* name, size_t count, std::vector<float>* out) {
        const bool ok = ReadFloatFile(base + name, count, out);
        if (!ok) LOGE("KleinDoubleBlockEngine: bad or missing %s (expected %zu floats)", name, count);
        return ok;
    };
    const bool read_ok =
        read("img.bin", kImg, &img) && read("txt.bin", kTxt, &txt) &&
        read("pe.bin", kImgPe, &pe) && read("pe_ctx.bin", kTxtPe, &pe_ctx) &&
        read("img_mod1_shift.bin", kHidden, &img1s) && read("img_mod1_scale.bin", kHidden, &img1sc) &&
        read("img_mod1_gate.bin", kHidden, &img1g) && read("img_mod2_shift.bin", kHidden, &img2s) &&
        read("img_mod2_scale.bin", kHidden, &img2sc) && read("img_mod2_gate.bin", kHidden, &img2g) &&
        read("txt_mod1_shift.bin", kHidden, &txt1s) && read("txt_mod1_scale.bin", kHidden, &txt1sc) &&
        read("txt_mod1_gate.bin", kHidden, &txt1g) && read("txt_mod2_shift.bin", kHidden, &txt2s) &&
        read("txt_mod2_scale.bin", kHidden, &txt2sc) && read("txt_mod2_gate.bin", kHidden, &txt2g) &&
        read("img_out.bin", kImg, &ref_img) && read("txt_out.bin", kTxt, &ref_txt);
    if (!read_ok) return env->NewStringUTF("FAILED: could not read double-block reference files");
    pockettavern::KleinDoubleBlockEngine engine;
    if (!engine.Load(model_dir, dispatch_dir)) return env->NewStringUTF("FAILED: Load() returned false; see logcat");
    std::vector<float> out_img, out_txt;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = engine.forwardZeroCopyPooled(img, txt, pe, pe_ctx, img1s, img1sc, img1g, img2s, img2sc, img2g,
                                   txt1s, txt1sc, txt1g, txt2s, txt2sc, txt2g, &out_img, &out_txt);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    if (!ok || out_img.size() != ref_img.size() || out_txt.size() != ref_txt.size()) return env->NewStringUTF("FAILED: forward() returned false or output sizes differ; see logcat");
    auto summarize = [](const std::vector<float>& out, const std::vector<float>& ref, float* max_diff, double* mean_diff, double* mean_ref, bool* nan_inf) {
        *max_diff = 0.f; double sum_diff = 0., sum_ref = 0.; *nan_inf = false;
        for (size_t i = 0; i < out.size(); ++i) { if (std::isnan(out[i]) || std::isinf(out[i])) *nan_inf = true; const float d = std::fabs(out[i] - ref[i]); if (d > *max_diff) *max_diff = d; sum_diff += d; sum_ref += std::fabs(ref[i]); }
        *mean_diff = sum_diff / out.size(); *mean_ref = sum_ref / out.size();
    };
    float img_max, txt_max; double img_diff, img_ref_mean, txt_diff, txt_ref_mean; bool img_bad, txt_bad;
    summarize(out_img, ref_img, &img_max, &img_diff, &img_ref_mean, &img_bad);
    summarize(out_txt, ref_txt, &txt_max, &txt_diff, &txt_ref_mean, &txt_bad);
    char buf[512];
    snprintf(buf, sizeof(buf), "OK elapsedMs=%lld hasNanOrInf=%d img(maxAbsDiff=%f meanAbsDiff=%f meanAbsRef=%f) txt(maxAbsDiff=%f meanAbsDiff=%f meanAbsRef=%f)", static_cast<long long>(elapsed), img_bad || txt_bad, img_max, img_diff, img_ref_mean, txt_max, txt_diff, txt_ref_mean);
    LOGE("KleinDoubleBlockEngine smoke test: %s", buf);
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinZeroCopyQkvFlashProbe(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir, jstring refDir) {
    const std::string model_dir = jstringToStd(env, modelDir);
    const std::string dispatch_dir = jstringToStd(env, dispatchLibDir);
    const std::string base = jstringToStd(env, refDir) + "/";
    std::vector<float> img, pe, shift, scale;
    if (!ReadFloatFile(base + "img.bin", 4096 * 3072, &img) ||
        !ReadFloatFile(base + "pe.bin", 4096 * 256, &pe) ||
        !ReadFloatFile(base + "img_mod1_shift.bin", 3072, &shift) ||
        !ReadFloatFile(base + "img_mod1_scale.bin", 3072, &scale)) {
        return env->NewStringUTF("FAILED: could not read zero-copy probe inputs");
    }
    pockettavern::KleinDoubleBlockEngine engine;
    if (!engine.Load(model_dir, dispatch_dir)) return env->NewStringUTF("FAILED: Load() returned false; see logcat");
    long long run_ms = -1; float max_diff = -1.f;
    if (!engine.RunZeroCopyQkvToFlashProbe(img, pe, shift, scale, &run_ms, &max_diff)) {
        return env->NewStringUTF("FAILED: zero-copy run rejected or mismatched; see logcat");
    }
    char buf[192];
    snprintf(buf, sizeof(buf), "OK directRunMs=%lld maxAbsDiff=%f", run_ms, max_diff);
    LOGE("Klein zero-copy qkv->flash probe: %s", buf);
    return env->NewStringUTF(buf);
}

// First component-level integration check.  time_in is used every denoising step and has a small
// fixed [1,256] input, so this validates model loading, NPU dispatch, and float32 DmaBuf copying
// before a full run allocates the substantially larger token and VAE activations.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinTimeInSmoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir) {
    const std::string model_dir = jstringToStd(env, modelDir);
    pockettavern::KleinComponentEngine engine;
    if (!engine.Load(jstringToStd(env, dispatchLibDir))) {
        return env->NewStringUTF("FAILED: component environment Load() returned false; see logcat");
    }
    std::vector<float> timestep_embedding(256);
    for (size_t i = 0; i < timestep_embedding.size(); ++i) {
        timestep_embedding[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.01));
    }
    std::vector<float> output;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = engine.Run(model_dir + "/time_in_Google_Tensor_G5.tflite", {&timestep_embedding},
                               {{1, 256}}, {1, 3072}, &output);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (!ok || output.size() != 3072) {
        return env->NewStringUTF("FAILED: time_in component execution failed; see logcat");
    }
    bool invalid = false;
    float max_abs = 0.0f;
    for (float value : output) {
        invalid = invalid || std::isnan(value) || std::isinf(value);
        max_abs = std::max(max_abs, std::fabs(value));
    }
    char buf[192];
    snprintf(buf, sizeof(buf), "OK elapsedMs=%lld outputSize=%zu hasNanOrInf=%d maxAbs=%f out[0..3]=%f,%f,%f,%f",
             static_cast<long long>(elapsed), output.size(), invalid, max_abs,
             output[0], output[1], output[2], output[3]);
    LOGE("Klein time_in component smoke: %s", buf);
    return env->NewStringUTF(buf);
}

// Runs the three fixed-shape component boundaries with the largest practical activation classes:
// image tokens, Qwen context tokens, and the VAE decoder's first stage.  They remain separate
// deliberately: each Run() tears down its compiled model/DmaBufs before the next begins.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinComponentBoundarySmoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir) {
    const std::string model_dir = jstringToStd(env, modelDir);
    pockettavern::KleinComponentEngine engine;
    if (!engine.Load(jstringToStd(env, dispatchLibDir))) {
        return env->NewStringUTF("FAILED: component environment Load() returned false; see logcat");
    }
    auto values = [](size_t count) {
        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(std::sin(i * 0.001) * 0.1);
        return result;
    };
    auto run = [&](const char* file, std::vector<float>* input, const std::vector<int32_t>& input_shape,
                   const std::vector<int32_t>& output_shape, std::vector<float>* output, long long* elapsed) {
        const auto start = std::chrono::steady_clock::now();
        const bool ok = engine.Run(model_dir + "/" + file, {input}, {input_shape}, output_shape, output);
        *elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        return ok;
    };
    auto img_input = values(4096 * 128);
    std::vector<float> img_output;
    long long img_ms = -1;
    if (!run("img_in_Google_Tensor_G5.tflite", &img_input, {1, 4096, 128}, {1, 4096, 3072},
             &img_output, &img_ms)) return env->NewStringUTF("FAILED: img_in component execution failed; see logcat");
    img_input.clear(); img_input.shrink_to_fit(); img_output.clear(); img_output.shrink_to_fit();

    auto txt_input = values(512 * 7680);
    std::vector<float> txt_output;
    long long txt_ms = -1;
    if (!run("txt_in_Google_Tensor_G5.tflite", &txt_input, {1, 512, 7680}, {1, 512, 3072},
             &txt_output, &txt_ms)) return env->NewStringUTF("FAILED: txt_in component execution failed; see logcat");
    txt_input.clear(); txt_input.shrink_to_fit(); txt_output.clear(); txt_output.shrink_to_fit();

    auto latent_input = values(128 * 64 * 64);
    std::vector<float> vae_output;
    long long vae_ms = -1;
    if (!run("vae_decoder_pre_mid_Google_Tensor_G5.tflite", &latent_input, {1, 128, 64, 64},
             {1, 512, 128, 128}, &vae_output, &vae_ms)) {
        return env->NewStringUTF("FAILED: VAE pre_mid component execution failed; see logcat");
    }
    bool invalid = false;
    for (float value : vae_output) invalid = invalid || std::isnan(value) || std::isinf(value);
    char buf[256];
    snprintf(buf, sizeof(buf), "OK imgInMs=%lld txtInMs=%lld vaePreMidMs=%lld vaeOutputSize=%zu hasNanOrInf=%d",
             img_ms, txt_ms, vae_ms, vae_output.size(), invalid);
    LOGE("Klein component-boundary smoke: %s", buf);
    return env->NewStringUTF(buf);
}

// Extends the VAE validation through the first spatial upsample. Its [1,512,256,256] output is
// 128 MiB in float32, large enough to exercise real decoder allocation behavior while avoiding
// the 512 MiB and 1 GiB activation boundaries of later stages.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinVaeUp3Smoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir) {
    const std::string model_dir = jstringToStd(env, modelDir);
    pockettavern::KleinComponentEngine engine;
    if (!engine.Load(jstringToStd(env, dispatchLibDir))) {
        return env->NewStringUTF("FAILED: component environment Load() returned false; see logcat");
    }
    std::vector<float> latents(128 * 64 * 64);
    for (size_t i = 0; i < latents.size(); ++i) latents[i] = static_cast<float>(std::sin(i * 0.001) * 0.1);
    std::vector<float> pre_mid;
    const auto pre_start = std::chrono::steady_clock::now();
    if (!engine.Run(model_dir + "/vae_decoder_pre_mid_Google_Tensor_G5.tflite", {&latents},
                    {{1, 128, 64, 64}}, {1, 512, 128, 128}, &pre_mid)) {
        return env->NewStringUTF("FAILED: VAE pre_mid execution failed; see logcat");
    }
    const auto pre_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - pre_start).count();
    latents.clear(); latents.shrink_to_fit();
    std::vector<float> up3;
    const auto up3_start = std::chrono::steady_clock::now();
    if (!engine.Run(model_dir + "/vae_decoder_up_3_Google_Tensor_G5.tflite", {&pre_mid},
                    {{1, 512, 128, 128}}, {1, 512, 256, 256}, &up3)) {
        return env->NewStringUTF("FAILED: VAE up_3 execution failed; see logcat");
    }
    const auto up3_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - up3_start).count();
    bool invalid = false;
    for (float value : up3) invalid = invalid || std::isnan(value) || std::isinf(value);
    char buf[224];
    snprintf(buf, sizeof(buf), "OK preMidMs=%lld up3Ms=%lld outputSize=%zu hasNanOrInf=%d",
             static_cast<long long>(pre_ms), static_cast<long long>(up3_ms), up3.size(), invalid);
    LOGE("Klein VAE up_3 smoke: %s", buf);
    return env->NewStringUTF(buf);
}

// Executes the real 5-double + 20-single transformer sequence with correctly shaped synthetic
// projections. This is intentionally a pipeline-memory test rather than an image test: prompt
// conditioning, RoPE positions, and modulation are supplied by later pipeline work.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinTransformerSmoke(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir) {
    constexpr size_t kHidden = 3072;
    constexpr size_t kImageTokens = 4096;
    constexpr size_t kTextTokens = 512;
    constexpr size_t kPeStride = 256;
    auto values = [](size_t count, float scale) {
        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(std::sin(i * 0.001) * scale);
        return result;
    };
    std::vector<float> img = values(kImageTokens * kHidden, 0.01f);
    std::vector<float> txt = values(kTextTokens * kHidden, 0.01f);
    std::vector<float> pe = values(kImageTokens * kPeStride, 0.01f);
    std::vector<float> pe_ctx = values(kTextTokens * kPeStride, 0.01f);
    std::array<pockettavern::KleinDoubleModulation, 5> double_modulations;
    std::array<pockettavern::KleinSingleModulation, 20> single_modulations;
    auto zero_modulation = [&] {
        pockettavern::KleinSingleModulation modulation;
        modulation.shift.assign(kHidden, 0.0f);
        modulation.scale.assign(kHidden, 0.0f);
        modulation.gate.assign(kHidden, 0.0f);
        return modulation;
    };
    for (auto& modulation : double_modulations) {
        modulation.image_first = zero_modulation();
        modulation.image_second = zero_modulation();
        modulation.text_first = zero_modulation();
        modulation.text_second = zero_modulation();
    }
    for (auto& modulation : single_modulations) modulation = zero_modulation();
    pockettavern::KleinTransformerEngine transformer;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = transformer.Forward(jstringToStd(env, modelDir), jstringToStd(env, dispatchLibDir),
                                        pe, pe_ctx, double_modulations, single_modulations,
                                        &img, &txt, /*single_worker_count=*/4);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (!ok || img.size() != kImageTokens * kHidden || txt.size() != kTextTokens * kHidden) {
        return env->NewStringUTF("FAILED: full transformer forward failed; see logcat");
    }
    bool invalid = false;
    for (float value : img) invalid = invalid || std::isnan(value) || std::isinf(value);
    for (float value : txt) invalid = invalid || std::isnan(value) || std::isinf(value);
    char buf[256];
    snprintf(buf, sizeof(buf), "OK elapsedMs=%lld imgSize=%zu txtSize=%zu hasNanOrInf=%d",
             static_cast<long long>(elapsed), img.size(), txt.size(), invalid);
    LOGE("Klein full-transformer smoke: %s", buf);
    return env->NewStringUTF(buf);
}

// One real Flux denoising step using the checked-in Qwen reference context. This is deliberately
// prompt-fixed until Qwen runs on-device, but every denoiser operation uses the model's actual
// weights, timestep modulation, and four-axis RoPE positions.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinOneStepReference(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir, jstring outputFile, jint stepIndex) {
    constexpr size_t H = 3072, I = 4096, T = 512, C = 7680, P = 256;
    const std::string dir = jstringToStd(env, modelDir);
    std::vector<float> context;
    if (!ReadFloatFile(dir + "/context.bin", T * C, &context)) return env->NewStringUTF("FAILED: context.bin missing or invalid");
    auto sine = [](size_t n, float scale) { std::vector<float> v(n); for (size_t i=0;i<n;++i) v[i]=static_cast<float>(std::sin(i*.001)*scale); return v; };
    if (stepIndex < 0 || stepIndex >= 4) return env->NewStringUTF("FAILED: invalid four-step schedule index");
    const std::string output_path = jstringToStd(env, outputFile);
    std::vector<float> latent;
    if (stepIndex == 0) {
        std::mt19937 generator(12345);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        latent.resize(I * 128);
        for (float& value : latent) value = normal(generator);
    }
    else if (!ReadFloatFile(output_path, I * 128, &latent)) return env->NewStringUTF("FAILED: prior-step latent missing or invalid");
    std::vector<float> time(256);
    const double raw_t = 1.0 - 0.25 * stepIndex;
    // Match sampling.compute_empirical_mu(image_seq_len=4096, num_steps=4), not the
    // asymptotic 200-step value. The four-step Klein schedule needs the interpolated shift.
    const double m200 = 0.00016927 * I + 0.45666666;
    const double m10 = 8.73809524e-05 * I + 1.89833333;
    const double mu = m200 + (4.0 - 200.0) * (m200 - m10) / 190.0;
    const double e=std::exp(mu);
    const double shifted_t=e/(e+(1.0/raw_t-1.0));
    const double raw_next = raw_t - .25;
    const double shifted_next = raw_next == 0.0 ? 0.0 : e/(e+(1.0/raw_next-1.0));
    for (size_t i=0;i<128;++i) { const double omega=std::exp(-std::log(10000.0)*i/128.0); time[i]=std::cos(1000.0*shifted_t*omega); time[128+i]=std::sin(1000.0*shifted_t*omega); }
    auto positions = [](size_t tokens, bool image) {
        std::vector<float> pe(tokens * 256);
        for (size_t n=0;n<tokens;++n) {
            const int coords[4] = {0, image ? static_cast<int>(n / 64) : 0, image ? static_cast<int>(n % 64) : 0, image ? 0 : static_cast<int>(n)};
            // BUG FIX: this must be the per-token base offset (n*256), not 0 -- resetting to 0
            // here made every token's RoPE data overwrite pe[0..255], leaving every token but the
            // last as all-zero. A zero rotation matrix zeros out apply_rope's output, which is a
            // very plausible root cause of the "Q/K near-zero" NPU investigation in
            // docs/FLUX2_KLEIN_HANDOFF.md -- not a Tensor G5 hardware/compiler bug at all.
            size_t o=n*256;
            for (int axis=0;axis<4;++axis) for (int j=0;j<16;++j) { const double a=coords[axis]/std::pow(2000.0, 2.0*j/32.0); const float c=std::cos(a), s=std::sin(a); pe[o++]=c; pe[o++]=-s; pe[o++]=s; pe[o++]=c; }
        }
        return pe;
    };
    const std::vector<float> pe = positions(I, true), pe_ctx = positions(T, false);
    pockettavern::KleinComponentEngine components;
    if (!components.Load(jstringToStd(env, dispatchLibDir))) return env->NewStringUTF("FAILED: component Load() failed");
    std::vector<float> vec, img, txt, prediction;
    if (!components.Run(dir + "/time_in_Google_Tensor_G5.tflite", {&time}, {{1,256}}, {1,H}, &vec) ||
        !components.Run(dir + "/img_in_Google_Tensor_G5.tflite", {&latent}, {{1,static_cast<int>(I),128}}, {1,static_cast<int>(I),static_cast<int>(H)}, &img) ||
        !components.Run(dir + "/txt_in_Google_Tensor_G5.tflite", {&context}, {{1,static_cast<int>(T),static_cast<int>(C)}}, {1,static_cast<int>(T),static_cast<int>(H)}, &txt)) return env->NewStringUTF("FAILED: input projection failed");
    std::vector<std::vector<float>> mi, mt, ms;
    const std::vector<std::vector<int32_t>> six(6, {1,static_cast<int>(H)}), three(3, {1,static_cast<int>(H)});
    if (!components.RunMulti(dir + "/mod_img_Google_Tensor_G5.tflite", {&vec}, {{1,static_cast<int>(H)}}, six, &mi) ||
        !components.RunMulti(dir + "/mod_txt_Google_Tensor_G5.tflite", {&vec}, {{1,static_cast<int>(H)}}, six, &mt) ||
        !components.RunMulti(dir + "/mod_single_Google_Tensor_G5.tflite", {&vec}, {{1,static_cast<int>(H)}}, three, &ms)) return env->NewStringUTF("FAILED: modulation projection failed");
    if (stepIndex == 0) {
        WriteFloatFile(dir + "/debug_latent.bin", latent);
        WriteFloatFile(dir + "/debug_time_vec.bin", vec);
        WriteFloatFile(dir + "/debug_img_in.bin", img);
        WriteFloatFile(dir + "/debug_txt_in.bin", txt);
        WriteFloatFile(dir + "/debug_mod_img_0.bin", mi[0]);
        WriteFloatFile(dir + "/debug_mod_img_1.bin", mi[1]);
        WriteFloatFile(dir + "/debug_mod_txt_0.bin", mt[0]);
        WriteFloatFile(dir + "/debug_mod_txt_1.bin", mt[1]);
        WriteFloatFile(dir + "/debug_mod_single_0.bin", ms[0]);
    }
    std::array<pockettavern::KleinDoubleModulation,5> dm;
    std::array<pockettavern::KleinSingleModulation,20> sm;
    for (auto& m:dm) { m.image_first={mi[0],mi[1],mi[2]}; m.image_second={mi[3],mi[4],mi[5]}; m.text_first={mt[0],mt[1],mt[2]}; m.text_second={mt[3],mt[4],mt[5]}; }
    for (auto& m:sm) m={ms[0],ms[1],ms[2]};
    if (stepIndex == 0) {
        pockettavern::KleinDoubleBlockEngine probe;
        std::vector<float> probe_img, probe_txt;
        std::vector<std::vector<float>> qkv_img, qkv_txt;
        const auto& m = dm[0];
        if (!probe.Load(dir, jstringToStd(env, dispatchLibDir), 0) ||
            !probe.DebugFirstQkv(img, txt, pe, pe_ctx, m.image_first.shift, m.image_first.scale,
                                 m.text_first.shift, m.text_first.scale, &qkv_img, &qkv_txt) ||
            !probe.forward(img, txt, pe, pe_ctx,
                m.image_first.shift, m.image_first.scale, m.image_first.gate,
                m.image_second.shift, m.image_second.scale, m.image_second.gate,
                m.text_first.shift, m.text_first.scale, m.text_first.gate,
                m.text_second.shift, m.text_second.scale, m.text_second.gate,
                &probe_img, &probe_txt)) return env->NewStringUTF("FAILED: double-block trace probe failed");
        WriteFloatFile(dir + "/debug_double0_img_q.bin", qkv_img[0]);
        WriteFloatFile(dir + "/debug_double0_img_k.bin", qkv_img[1]);
        WriteFloatFile(dir + "/debug_double0_img_v.bin", qkv_img[2]);
        WriteFloatFile(dir + "/debug_double0_txt_q.bin", qkv_txt[0]);
        WriteFloatFile(dir + "/debug_double0_txt_k.bin", qkv_txt[1]);
        WriteFloatFile(dir + "/debug_double0_txt_v.bin", qkv_txt[2]);
        WriteFloatFile(dir + "/debug_double0_img.bin", probe_img);
        WriteFloatFile(dir + "/debug_double0_txt.bin", probe_txt);
    }
    pockettavern::KleinTransformerEngine transformer;
    const auto start=std::chrono::steady_clock::now();
    if (!transformer.Forward(dir, jstringToStd(env, dispatchLibDir), pe, pe_ctx, dm, sm, &img, &txt, 4) ||
        !components.Run(dir + "/final_Google_Tensor_G5.tflite", {&img,&vec}, {{1,static_cast<int>(I),static_cast<int>(H)},{1,static_cast<int>(H)}}, {1,static_cast<int>(I),128}, &prediction)) return env->NewStringUTF("FAILED: transformer or final layer failed");
    for (size_t i=0;i<latent.size();++i) latent[i]+=static_cast<float>((shifted_next-shifted_t)*prediction[i]);
    if (!WriteFloatFile(output_path, latent)) return env->NewStringUTF("FAILED: could not write updated latent");
    const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();
    char b[160]; snprintf(b,sizeof(b),"OK step=%d elapsedMs=%lld latentFloats=%zu",stepIndex,static_cast<long long>(elapsed),latent.size()); LOGE("Klein one-step reference: %s",b); return env->NewStringUTF(b);
}

// Direct, non-tiled staged VAE decode. Each Run() releases its compiled model and DmaBufs before
// the next stage, keeping the peak bounded to one stage's input/output pair.
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeDecodeKleinReferenceLatent(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring dispatchLibDir, jstring outputFile) {
    const std::string dir = jstringToStd(env, modelDir);
    std::vector<float> tokens;
    if (!ReadFloatFile(dir + "/one_step_latent.bin", 4096 * 128, &tokens)) return env->NewStringUTF("FAILED: final latent missing");
    std::vector<float> current(128 * 64 * 64);
    for (size_t h=0; h<64; ++h) for (size_t w=0; w<64; ++w) for (size_t c=0; c<128; ++c) current[(c*64+h)*64+w] = tokens[(h*64+w)*128+c];
    tokens.clear(); tokens.shrink_to_fit();
    pockettavern::KleinComponentEngine engine;
    if (!engine.Load(jstringToStd(env, dispatchLibDir))) return env->NewStringUTF("FAILED: VAE environment failed");
    struct Stage { const char* file; std::vector<int32_t> input; std::vector<int32_t> output; };
    const Stage stages[] = {
        {"vae_decoder_pre_mid_Google_Tensor_G5.tflite",{1,128,64,64},{1,512,128,128}}, {"vae_decoder_up_3_Google_Tensor_G5.tflite",{1,512,128,128},{1,512,256,256}},
        {"vae_decoder_up_2_Google_Tensor_G5.tflite",{1,512,256,256},{1,512,512,512}}, {"vae_decoder_up_1_Google_Tensor_G5.tflite",{1,512,512,512},{1,256,1024,1024}},
        {"vae_decoder_up_0_block_0_Google_Tensor_G5.tflite",{1,256,1024,1024},{1,128,1024,1024}}, {"vae_decoder_up_0_block_1_Google_Tensor_G5.tflite",{1,128,1024,1024},{1,128,1024,1024}},
        {"vae_decoder_up_0_block_2_Google_Tensor_G5.tflite",{1,128,1024,1024},{1,128,1024,1024}}, {"vae_decoder_up_0_head_Google_Tensor_G5.tflite",{1,128,1024,1024},{1,3,1024,1024}},
    };
    const auto start=std::chrono::steady_clock::now();
    for (const Stage& stage : stages) { std::vector<float> next; if (!engine.Run(dir + "/" + stage.file, {&current}, {stage.input}, stage.output, &next)) return env->NewStringUTF("FAILED: VAE stage failed; see logcat"); current.swap(next); }
    if (!WriteRgbPpm(jstringToStd(env, outputFile), current)) return env->NewStringUTF("FAILED: could not write decoded PPM");
    const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();
    char b[160]; snprintf(b,sizeof(b),"OK elapsedMs=%lld outputFloats=%zu",static_cast<long long>(elapsed),current.size()); LOGE("Klein direct VAE decode: %s",b); return env->NewStringUTF(b);
}

// FLUX.2 [klein] Phase 1 (docs/FLUX2_KLEIN_PHASE1_TEXT_ENCODER_PLAN.md): runs the on-device
// Qwen3-4B text encoder for an arbitrary prompt and dumps the resulting [1,512,7680] hidden-state
// tensor to outputFile in the same raw-float32 layout as the checked-in context.bin, so it can be
// pulled and diffed against scripts/export_klein_qwen_reference.py's PyTorch output the same way
// every other Klein component was validated. configPath points at the MNN-exported encoder's
// config.json (see docs/flux2-klein-conversion.md's 2026-08-30 entry for the export command).
JNIEXPORT jstring JNICALL
Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunQwenTextEncoder(
        JNIEnv* env, jclass /*clazz*/, jstring configPath, jstring prompt, jstring outputFile) {
    pockettavern::QwenTextEncoderEngine encoder;
    const auto start = std::chrono::steady_clock::now();
    if (!encoder.Load(jstringToStd(env, configPath), /*mmap_cache_dir=*/"")) {
        return env->NewStringUTF("FAILED: could not load Qwen text encoder; see logcat");
    }
    std::vector<float> hidden_states;
    if (!encoder.Encode(jstringToStd(env, prompt), &hidden_states)) {
        return env->NewStringUTF("FAILED: encode failed; see logcat");
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (!WriteFloatFile(jstringToStd(env, outputFile), hidden_states)) {
        return env->NewStringUTF("FAILED: could not write hidden_states output");
    }
    bool invalid = false;
    for (float value : hidden_states) invalid = invalid || std::isnan(value) || std::isinf(value);
    char buf[160];
    snprintf(buf, sizeof(buf), "OK elapsedMs=%lld hiddenStatesFloats=%zu hasNanOrInf=%d",
             static_cast<long long>(elapsed), hidden_states.size(), invalid);
    LOGE("Qwen text encoder: %s", buf);
    return env->NewStringUTF(buf);
}

// Production end-to-end FLUX.2 [klein] generation: arbitrary prompt + seed -> real PNG. Loads a
// fresh KleinDiffusionEngine per call (matches this pipeline's existing diagnostic architecture,
// which already reloads everything per invocation -- no persistent native handle to manage).
// Blocking -- must be called from a background thread; progressCallback fires synchronously on
// the calling thread across the pipeline's 13 phases (encode, 4 denoise steps, 8 VAE stages).
JNIEXPORT jboolean JNICALL
Java_com_pockettavern_app_data_local_inference_KleinDiffusionBridge_nativeGenerate(
        JNIEnv* env, jobject /*thiz*/, jstring npuModelDir, jstring dispatchLibDir,
        jstring qwenConfigPath, jstring mmapCacheDir, jstring prompt, jstring outputPngPath,
        jint seed, jobject progressCallback) {
    pockettavern::KleinDiffusionEngine engine;
    if (!engine.Load(jstringToStd(env, npuModelDir), jstringToStd(env, dispatchLibDir),
                      jstringToStd(env, qwenConfigPath), jstringToStd(env, mmapCacheDir))) {
        return JNI_FALSE;
    }
    KotlinProgressCallback callback(env, progressCallback);
    const bool ok = engine.Generate(jstringToStd(env, prompt), static_cast<uint32_t>(seed),
                                     jstringToStd(env, outputPngPath), std::ref(callback));
    return ok ? JNI_TRUE : JNI_FALSE;
}

}  // extern "C"
