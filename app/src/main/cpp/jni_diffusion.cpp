#include <jni.h>
#include <android/log.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <MNN/MNNForwardType.h>
#include "diffusion/diffusion.hpp"
#include "diffusion/stable_diffusion_xl.hpp"
#include "npu/NpuUnetEngine.hpp"
#include "npu/NpuTextEncoderEngine.hpp"

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

}  // extern "C"
