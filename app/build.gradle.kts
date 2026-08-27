import java.util.Properties
import org.gradle.api.tasks.Sync

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.hilt)
    alias(libs.plugins.ksp)
}

val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) load(f.inputStream())
}

// CMake needs libLiteRt.so for native link-time resolution. Resolve the same declared AAR that
// packages LiteRT into the APK and extract its jni/ directory under build/, rather than relying
// on AGP's content-hash-named transform cache.
val liteRtNativeAar = configurations.detachedConfiguration(
    dependencies.create("com.google.ai.edge.litert:litert:2.2.0@aar")
)
val liteRtJniDirectory = layout.buildDirectory.dir("generated/litert-jni")
val extractLiteRtNative by tasks.registering(Sync::class) {
    from({ zipTree(liteRtNativeAar.singleFile) }) {
        include("jni/**")
        eachFile { path = path.removePrefix("jni/") }
        includeEmptyDirs = false
    }
    into(liteRtJniDirectory)
}

// Native CMake configuration can run before preBuild in a task graph, so make every configure
// task explicitly depend on the extracted library it receives through LITERT_JNI_DIR.
tasks.configureEach {
    if (name.startsWith("configureCMake")) dependsOn(extractLiteRtNative)
}

android {
    namespace = "com.pockettavern.app"
    compileSdk = 36  // required by Llamatik (llama.cpp GGUF); targetSdk stays 35
    // AGP 8.7.3's default ndkVersion (27.0.12077973) conflicts with the NDK actually installed
    // and used for the MNN native build (r29, /opt/android-ndk) -- pin explicitly to match.
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "com.pockettavern.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 23
        versionName = "2.3.2"

        // Stories (native ensemble) = private/dev feature for now. Visible in debug builds,
        // hidden in the public release (overridden false below). Keeps PocketTavern simple.
        buildConfigField("boolean", "STORIES_ENABLED", "true")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Phones are arm64; dropping x86_64 ~halves the APK (large on-device native libs).
        ndk { abiFilters += "arm64-v8a" }

        externalNativeBuild {
            cmake {
                // c++_shared, not c++_static: PocketTavern already ships other native libs as
                // prebuilt AARs (LiteRT-LM, Llamatik) -- statically linking libc++ into more
                // than one .so in the same process is an NDK-documented hazard (duplicate
                // global state / ODR issues). See app/src/main/cpp/CMakeLists.txt's comment.
                arguments += "-DANDROID_STL=c++_shared"
                // AGP maps the Gradle "debug" build variant to CMAKE_BUILD_TYPE=Debug by default
                // (no -O optimization) unless told otherwise -- confirmed via
                // app/.cxx/Debug/*/arm64-v8a/CMakeCache.txt showing CMAKE_BUILD_TYPE=Debug, vs.
                // the validated-fast CLI reference build's CMAKE_BUILD_TYPE=Release. On-device
                // this was ~170s/UNet-step instead of the CLI's ~30-33s/step -- a ~5x gap in line
                // with an unoptimized debug build of compute-heavy tensor math, not just a
                // scheduling/affinity difference (see Power_High comment in
                // stable_diffusion_xl.cpp, which addressed a real but much smaller effect).
                // Force Release for the native side regardless of the app's own debug/release
                // variant -- there's no reason to ship or test an unoptimized libMNN.so.
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DLITERT_JNI_DIR=${liteRtJniDirectory.get().asFile.absolutePath}"
                // Without this, Ninja builds every default target CMake defines -- MNN_BUILD_LLM
                // pulls in a pile of demo/tool executables (llm_demo, llm_bench, embedding_demo,
                // quantize_llm, etc.) regardless of MNN_BUILD_TOOLS/MNN_BUILD_DEMO, none of which
                // the app needs. pockettavern_diffusion links against MNN, which pulls it in
                // transitively -- nothing else needs to be built.
                targets += "pockettavern_diffusion"
            }
        }
    }

    signingConfigs {
        create("release") {
            storeFile = file("pockettavern.keystore")
            storePassword = localProps.getProperty("KEYSTORE_PASSWORD", "")
            keyAlias = localProps.getProperty("KEY_ALIAS", "pockettavern")
            keyPassword = localProps.getProperty("KEY_PASSWORD", "")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            buildConfigField("boolean", "STORIES_ENABLED", "false")  // hide Stories in the public release
            signingConfig = signingConfigs.getByName("release")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        // litertlm-android ships newer Kotlin metadata (2.3.x) than our compiler (2.1.x).
        // Safe to consume — it's a JNI-wrapper lib with simple public types.
        freeCompilerArgs += "-Xskip-metadata-version-check"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    // litert:2.2.0 and litertlm-android:0.13.1 both bundle a same-named but DIFFERENT-content
    // libLiteRt.so (only litert:2.2.0's implements the CompiledModel/NPU JNI surface litert-api
    // needs). pickFirst can't target "which AAR" -- verified by inspecting the merged APK's
    // extracted libLiteRt.so directly, don't assume.
    packaging {
        jniLibs {
            pickFirsts += "**/libLiteRt*.so"
            // Without this, native libs are mmap'd uncompressed straight out of the APK zip
            // instead of extracted to a real lib/arm64 directory on disk -- LiteRT's NPU dispatch
            // loader does a literal directory scan for libLiteRtDispatch_*.so and finds nothing
            // without it (confirmed: google-ai-edge/litert-samples' NPU sample carries the same
            // flag with the same "needed for NPU runtimes" comment).
            useLegacyPackaging = true
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            // Pinned rather than left to AGP's default so the SDK-managed cmake/ninja package
            // Gradle downloads is predictable. 3.22.1 (a common AGP default) failed to configure
            // MNN with MNN_SEP_BUILD=OFF: CMake rejected add_custom_command(TARGET ... PRE_BUILD)
            // calls on the "MNNOpenCV"/"llm" OBJECT-library targets (tools/cv/CMakeLists.txt,
            // transformers/llm/engine/CMakeLists.txt) -- the same config built fine with the
            // system cmake (4.4.2) used for the desktop/Android CLI builds validated earlier
            // this session, so this looks like version-specific enforcement of that restriction.
            // Using the newest SDK-managed version to get closer to what actually worked.
            version = "4.1.2"
        }
    }
}

dependencies {
    // Core
    implementation(libs.androidx.core.ktx)
    implementation("androidx.core:core-splashscreen:1.0.1")
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)

    // Compose
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material3.adaptive)
    implementation(libs.androidx.material3.adaptive.layout)
    implementation(libs.androidx.material3.adaptive.navigation)
    implementation(libs.androidx.material.icons.extended)
    debugImplementation(libs.androidx.ui.tooling)

    // Navigation
    implementation(libs.androidx.navigation.compose)

    // Hilt
    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)
    implementation(libs.androidx.hilt.navigation.compose)

    // Network
    implementation(libs.retrofit)
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.retrofit.kotlinx.serialization)

    // On-device inference (LiteRT-LM, Apache-2.0). minSdk 23, arm64-v8a/x86_64.
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.13.1")
    // On-device GGUF inference via llama.cpp (Llamatik, MIT). Unlocks the GGUF ecosystem.
    implementation("com.llamatik:library:1.8.0")
    // NPU (Google Tensor / Darwinn) AOT-compiled model execution via the official CompiledModel
    // API. litert-api carries the Kotlin surface (CompiledModel, Environment, Accelerator,
    // NpuAcceleratorProvider); litert carries the native libLiteRt.so implementing it -- they're
    // separate Maven artifacts and litert-api does NOT pull litert in transitively (checked its
    // .pom: only androidx.lifecycle/guava/coroutines-guava/play:ai-delivery).
    implementation("com.google.ai.edge.litert:litert:2.2.0")
    implementation("com.google.ai.edge.litert:litert-api:2.2.0")

    // DataStore
    implementation(libs.androidx.datastore.preferences)

    // Image Loading
    implementation(libs.coil.compose)
    implementation(libs.coil.gif)

    // Chrome Custom Tabs for OAuth
    implementation("androidx.browser:browser:1.8.0")

    // Room database (character/chat index)
    val roomVersion = "2.6.1"
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    // SAF DocumentFile support (for folder import)
    implementation("androidx.documentfile:documentfile:1.0.1")

    // Encrypted storage for API keys and tokens
    implementation("androidx.security:security-crypto:1.0.0")
}
