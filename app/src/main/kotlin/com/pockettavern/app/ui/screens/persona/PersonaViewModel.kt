package com.pockettavern.app.ui.screens.persona

import android.content.Context
import android.util.Base64
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.pockettavern.app.data.local.SettingsDataStore
import com.pockettavern.app.data.repository.ImageGenRepository
import com.pockettavern.app.data.repository.LocalRepository
import com.pockettavern.app.data.repository.SettingsRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import com.pockettavern.app.domain.model.ForgeGenerationParams
import com.pockettavern.app.domain.model.GenerationState
import com.pockettavern.app.domain.model.ImageGenCapabilities
import com.pockettavern.app.domain.model.Persona
import com.pockettavern.app.domain.model.PersonaPosition
import com.pockettavern.app.domain.model.PersonaRole
import com.pockettavern.app.domain.model.Result
import com.pockettavern.app.domain.model.UserPersona
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class PersonaUiState(
    val personas: List<Persona> = emptyList(),
    val selectedPersona: Persona? = null,
    val serverUrl: String = "",
    val isLoading: Boolean = false,
    val isSaving: Boolean = false,
    val showEditDialog: Boolean = false,
    val editingPersona: Persona? = null,
    val editDescription: String = "",
    val editPosition: PersonaPosition = PersonaPosition.IN_PROMPT,
    val editRole: PersonaRole = PersonaRole.SYSTEM,
    val editDepth: Int = 2,
    val showDeleteConfirm: Boolean = false,
    val showCreateDialog: Boolean = false,
    // Shared by both the Create and Edit dialogs -- only one is ever open at a time, and both
    // need the same picker/generate flow for the persona's one avatar image.
    val avatarImageBytes: ByteArray? = null,
    val avatarImageMimeType: String = "image/png",
    val createName: String = "",
    val createDescription: String = "",
    val forgeAvailable: Boolean = false,
    val imageGenCapabilities: ImageGenCapabilities = ImageGenCapabilities(),
    val generationPrompt: String = "",
    val isGenerating: Boolean = false,
    val generationProgress: Float = 0f,
    val error: String? = null,
    val successMessage: String? = null
)

@HiltViewModel
class PersonaViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val localRepository: LocalRepository,
    private val settingsRepository: SettingsRepository,
    private val imageGenRepository: ImageGenRepository,
    private val settingsDataStore: SettingsDataStore
) : ViewModel() {

    private val _uiState = MutableStateFlow(PersonaUiState())
    val uiState: StateFlow<PersonaUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null

    init {
        viewModelScope.launch {
            val settings = settingsRepository.getSettings()
            val capabilities = imageGenRepository.getCapabilities()
            _uiState.update {
                it.copy(forgeAvailable = settings.imageGenBackendConfigured, imageGenCapabilities = capabilities)
            }
        }
        loadPersonas()
    }

    fun loadPersonas() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true) }
            when (val result = localRepository.getUserPersona()) {
                is Result.Success -> {
                    val persona = result.data.toPersona()
                    _uiState.update {
                        it.copy(
                            personas = listOf(persona),
                            selectedPersona = persona,
                            isLoading = false
                        )
                    }
                }
                is Result.Error -> {
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            error = result.exception.message
                        )
                    }
                }
            }
        }
    }

    fun selectPersona(persona: Persona) {
        // Only one persona in standalone mode — selecting it is a no-op
        _uiState.update { it.copy(selectedPersona = persona) }
    }

    fun showEditDialog(persona: Persona) {
        // Preload the current avatar's bytes so the picker shows what's actually saved, not
        // blank -- savePersonaEdit() always writes avatarImageBytes back out, so leaving this
        // null here would silently drop the avatar again the moment the user hits Save without
        // touching the picker.
        val avatarBytes = persona.avatarId.takeIf { it.isNotBlank() }
            ?.let { path -> runCatching { File(path).readBytes() }.getOrNull() }
        _uiState.update {
            it.copy(
                showEditDialog = true,
                editingPersona = persona,
                editDescription = persona.description,
                editPosition = persona.position,
                editRole = persona.role,
                editDepth = persona.depth,
                avatarImageBytes = avatarBytes,
                generationPrompt = "",
                isGenerating = false,
                generationProgress = 0f
            )
        }
    }

    fun hideEditDialog() {
        _uiState.update {
            it.copy(
                showEditDialog = false,
                editingPersona = null,
                avatarImageBytes = null
            )
        }
    }

    fun updateEditDescription(description: String) {
        _uiState.update { it.copy(editDescription = description) }
    }

    fun updateEditPosition(position: PersonaPosition) {
        _uiState.update { it.copy(editPosition = position) }
    }

    fun updateEditRole(role: PersonaRole) {
        _uiState.update { it.copy(editRole = role) }
    }

    fun updateEditDepth(depth: Int) {
        _uiState.update { it.copy(editDepth = depth.coerceIn(0, 100)) }
    }

    fun savePersonaEdit() {
        val state = _uiState.value
        val persona = state.editingPersona ?: return

        viewModelScope.launch {
            _uiState.update { it.copy(isSaving = true) }
            try {
                // Copy forward from the currently-stored persona, not a fresh UserPersona(...) --
                // noSpeakForUser (a field this dialog has no control for) must carry over
                // unchanged. Building a fresh UserPersona() here previously silently reset it (and
                // avatarPath) to defaults on every single edit, discarding a real, already-
                // generated avatar file even though the file itself was never touched -- confirmed
                // the file survives on disk, only the DataStore reference was lost.
                val current = (localRepository.getUserPersona() as? Result.Success)?.data
                // Always rewritten from avatarImageBytes (preloaded from the existing file in
                // showEditDialog() if the user doesn't touch the picker) -- see that function's
                // comment for why leaving this unconditional matters.
                val avatarPath = state.avatarImageBytes?.let { bytes ->
                    val file = File(context.filesDir, "persona_avatar.png")
                    file.writeBytes(bytes)
                    file.absolutePath
                } ?: current?.avatarPath
                val updated = (current ?: UserPersona(name = persona.name)).copy(
                    name = persona.name,
                    description = state.editDescription,
                    position = state.editPosition.value,
                    depth = state.editDepth,
                    role = state.editRole.value,
                    avatarPath = avatarPath
                )
                localRepository.saveUserPersona(updated)
                _uiState.update {
                    it.copy(
                        isSaving = false,
                        showEditDialog = false,
                        avatarImageBytes = null,
                        successMessage = "Persona updated"
                    )
                }
                loadPersonas()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isSaving = false,
                        error = e.message
                    )
                }
            }
        }
    }

    fun showDeleteConfirm() {
        _uiState.update { it.copy(showDeleteConfirm = true) }
    }

    fun hideDeleteConfirm() {
        _uiState.update { it.copy(showDeleteConfirm = false) }
    }

    fun deletePersona() {
        // Deleting the only persona is not allowed in standalone mode
        _uiState.update {
            it.copy(
                showDeleteConfirm = false,
                showEditDialog = false,
                error = "Cannot delete the only persona in standalone mode"
            )
        }
    }

    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }

    fun clearSuccess() {
        _uiState.update { it.copy(successMessage = null) }
    }

    fun showCreateDialog() {
        _uiState.update {
            it.copy(
                showCreateDialog = true,
                avatarImageBytes = null,
                avatarImageMimeType = "image/png",
                createName = "",
                createDescription = "",
                generationPrompt = "",
                isGenerating = false,
                generationProgress = 0f
            )
        }
    }

    fun hideCreateDialog() {
        _uiState.update {
            it.copy(
                showCreateDialog = false,
                avatarImageBytes = null
            )
        }
    }

    fun setAvatarImage(bytes: ByteArray, mimeType: String) {
        _uiState.update {
            it.copy(
                avatarImageBytes = bytes,
                avatarImageMimeType = mimeType
            )
        }
    }

    fun updateCreateName(name: String) {
        _uiState.update { it.copy(createName = name) }
    }

    fun updateCreateDescription(description: String) {
        _uiState.update { it.copy(createDescription = description) }
    }

    fun createPersona() {
        val state = _uiState.value
        if (state.createName.isBlank()) {
            _uiState.update { it.copy(error = "Please enter a name") }
            return
        }

        viewModelScope.launch {
            _uiState.update { it.copy(isSaving = true) }
            try {
                val avatarPath = state.avatarImageBytes?.let { bytes ->
                    val file = File(context.filesDir, "persona_avatar.png")
                    file.writeBytes(bytes)
                    file.absolutePath
                }
                // Copy forward position/depth/role/noSpeakForUser from whatever persona already
                // exists -- this is a single-persona app (see class doc), so "Create" here really
                // means "replace name/description/avatar," not "start over" (same reasoning as
                // savePersonaEdit()'s fix).
                val current = (localRepository.getUserPersona() as? Result.Success)?.data
                val updated = (current ?: UserPersona()).copy(
                    name = state.createName.trim(),
                    description = state.createDescription,
                    avatarPath = avatarPath ?: current?.avatarPath
                )
                localRepository.saveUserPersona(updated)
                _uiState.update {
                    it.copy(
                        isSaving = false,
                        showCreateDialog = false,
                        avatarImageBytes = null,
                        successMessage = "Persona updated to \"${state.createName.trim()}\""
                    )
                }
                loadPersonas()
            } catch (e: Exception) {
                _uiState.update { it.copy(isSaving = false, error = e.message) }
            }
        }
    }

    fun updateGenerationPrompt(prompt: String) {
        _uiState.update { it.copy(generationPrompt = prompt) }
    }

    fun generateImage() {
        val prompt = _uiState.value.generationPrompt
        if (prompt.isBlank()) {
            _uiState.update { it.copy(error = "Please enter a prompt") }
            return
        }

        generationJob?.cancel()
        generationJob = viewModelScope.launch {
            _uiState.update { it.copy(isGenerating = true, generationProgress = 0f) }

            val config = settingsDataStore.getImageGenConfig()
            val params = ForgeGenerationParams(
                prompt = prompt,
                negativePrompt = config.negativePrompt,
                width = config.width,
                height = config.height,
                steps = config.steps,
                cfgScale = config.cfgScale,
                sampler = config.sampler,
                seed = config.seed
            )

            imageGenRepository.generateImageWithProgress(params).collect { state ->
                when (state) {
                    is GenerationState.Starting -> {
                        _uiState.update { it.copy(generationProgress = 0f) }
                    }
                    is GenerationState.InProgress -> {
                        _uiState.update { it.copy(generationProgress = state.progress) }
                    }
                    is GenerationState.Complete -> {
                        val imageBytes = Base64.decode(state.imageBase64, Base64.DEFAULT)
                        _uiState.update {
                            it.copy(
                                isGenerating = false,
                                avatarImageBytes = imageBytes,
                                avatarImageMimeType = "image/png"
                            )
                        }
                    }
                    is GenerationState.Error -> {
                        _uiState.update {
                            it.copy(
                                isGenerating = false,
                                error = state.message
                            )
                        }
                    }
                    GenerationState.Idle -> {}
                }
            }
        }
    }

    fun cancelGeneration() {
        generationJob?.cancel()
        viewModelScope.launch {
            imageGenRepository.interrupt()
            _uiState.update { it.copy(isGenerating = false, generationProgress = 0f) }
        }
    }

    // Convert UserPersona to Persona (domain model used by the UI)
    private fun UserPersona.toPersona(): Persona = Persona(
        avatarId = avatarPath ?: "",
        name = name,
        description = description,
        position = PersonaPosition.fromInt(position),
        role = PersonaRole.fromInt(role),
        depth = depth,
        isSelected = true
    )
}
