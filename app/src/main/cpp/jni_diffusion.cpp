#include <jni.h>
#include <android/log.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include <MNN/MNNForwardType.h>
#include "diffusion/diffusion.hpp"
#include "diffusion/stable_diffusion_xl.hpp"
#include "npu/NpuUnetEngine.hpp"

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
            LOGE("NPU UNet was requested for unsupported diffusion modelType=%d", modelType);
            delete diffusion;
            return 0;
        }
        auto* sdxl = static_cast<StableDiffusionXL*>(diffusion);
        if (!sdxl->configureNpuUnet(std::move(npuPath), std::move(dispatchPath))) {
            LOGE("Failed to configure NPU UNet");
            delete diffusion;
            return 0;
        }
        LOGE("Configured LiteRT NPU UNet for SDXL");
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

}  // extern "C"
