# Three-Tier Configuration System Implementation

## Overview
Implemented a flexible three-tier configuration precedence system for API endpoints and ML models, allowing users to customize server endpoints, LLM endpoints, and models (Whisper, Diarization) at multiple levels.

## Configuration Precedence
1. 🔵 **Defaults** - Built-in codebase defaults (lowest priority)
2. 🟢 **Environment Variables** - Values from `.env` file (medium priority)  
3. 🟡 **UI Settings** - Custom values set in Settings page (highest priority)

## Key Components

### 1. ConfigManager (`src/config.py`)
Central configuration management class that:
- Defines default values for all configurable settings
- Loads environment variables from `.env` file
- Implements precedence logic via `get()` method
- Provides `get_display_value()` to show configuration source
- Helper methods: `is_using_default()`, `is_using_env()`, `is_using_ui()`

**Configurable Settings:**
- `API_BASE_URL`: Flask API server endpoint (default: `http://localhost:5001`)
- `LLM_API_BASE_URL`: LLM API endpoint for summarization (default: `https://api.openai.com/v1`)
- `LLM_MODEL`: LLM model name (default: `gpt-4`)
- `OPENAI_API_KEY`: API key for authentication (default: empty)
- `HUGGINGFACE_TOKEN`: Token for PyAnnote and HuggingFace models (default: empty)
- `WHISPER_MODEL`: Whisper transcription model (default: `base`)
  - Three-tier precedence: 1) API request option 2) Environment variable 3) "base" default
  - OpenAI models: `tiny`, `base`, `small`, `medium`, `large`
  - HuggingFace: `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo`, etc.
  - Local: absolute path like `/Users/name/models/whisper-large`
- `DIARIZATION_MODEL`: PyAnnote diarization model (default: `pyannote/speaker-diarization-3.1`)
  - HuggingFace: `pyannote/speaker-diarization-3.1`, `pyannote/speaker-diarization-2.1`
  - Local: absolute path like `/Users/name/models/diarization`

### 2. Session State Management (`src/ui/app_new.py`)
- Uses `ui_*` prefixed keys for UI overrides:
  - `ui_api_base_url`
  - `ui_llm_api_base_url`
  - `ui_llm_model`
  - `ui_llm_api_key`
  - `ui_whisper_model`
  - `ui_diarization_model`
- Empty string (`""`) means "not overridden by UI" → falls back to env/defaults
- Non-empty string means "UI override active"

### 3. Helper Functions
- `get_effective_config(key)`: Gets final resolved value for any config key
- `get_api_base_url()`: Returns effective API server URL
- `get_llm_config()`: Returns dict with LLM configuration (base_url, model, api_key)
- `get_whisper_model()`: Returns effective Whisper model name
- `get_diarization_model()`: Returns effective diarization model name

### 4. Enhanced Settings Page
- Visual indicators showing configuration source for each setting
- Smart placeholders showing fallback values
- Apply/Cancel buttons for configuration changes
- "Current Effective Configuration" summary section
- **Model Configuration Section**: Configure Whisper and Diarization models with source indicators

### 5. Model Support (`src/audio/`)
**AudioTranscriber** (`transcription.py`):
- Supports OpenAI Whisper models (tiny, base, small, medium, large)
- Supports HuggingFace models (e.g., `openai/whisper-large-v3`)
- Supports local model paths
- Auto-detects model type and uses appropriate loading mechanism
- Falls back to `base` model if custom model fails to load

**PyannoteDiarizer** (`diarization.py`):
- Accepts configurable `model_name` parameter
- Supports HuggingFace PyAnnote models
- Supports local model paths
- Default: `pyannote/speaker-diarization-3.1`

## Benefits

1. **Flexibility**: Users can override any configuration (endpoints, models) without modifying code or .env files
2. **Development/Production**: Easy switching between environments and models
3. **On-Premises Support**: Can use custom LLM endpoints (vLLM, Ollama, etc.) and custom models
4. **Transparency**: UI clearly shows where each value comes from
5. **Fallback Safety**: Empty UI fields automatically use env/default values
6. **Model Variety**: Support for OpenAI, HuggingFace, and local models for both transcription and diarization

## Files Modified

- `src/config.py` - New file with ConfigManager class
  - Added WHISPER_MODEL and DIARIZATION_MODEL to DEFAULTS
- `src/ui/app_new.py` - Updated to use three-tier configuration system
  - `initialize_session_state()` - Added ui_* keys for all configurations
  - `get_whisper_model()`, `get_diarization_model()` - New helper functions
  - `settings_page()` - Complete rewrite with visual indicators and model configuration
  - `recording_page()` - Updated to use configuration helpers and display current models
  - `jobs_page()`, `transcript_page()` - Updated to use helper functions
  - All API client initialization uses dynamic configuration
- `src/audio/transcription.py` - Enhanced AudioTranscriber
  - Supports OpenAI Whisper, HuggingFace, and local models
  - Auto-detection of model type
  - Graceful fallback on load failures
- `src/audio/diarization.py` - Enhanced PyannoteDiarizer
  - Accepts configurable model_name parameter
  - Supports HuggingFace and local models
- `src/server/processor.py` - Updated to use model configuration
  - Three-tier precedence for whisper_model: API option > WHISPER_MODEL env > "base" default
  - Lazy loads and caches transcriber when model changes
  - Passes diarization_model from options
- `README.md` - Documented new configuration system, model support, and client-server architecture
- `.github/IMPLEMENTATION_NOTES.md` - Technical documentation for developers

## Usage Examples

**Example 1: Use defaults (no configuration needed)**
```python
# Just run the app - uses http://localhost:5001 and OpenAI API
```

**Example 2: Configure via .env**
```bash
# .env
API_BASE_URL=http://production-server:5001
LLM_API_BASE_URL=https://custom-llm.company.com/v1
LLM_MODEL=gpt-4-turbo
OPENAI_API_KEY=sk-...
```

**Example 3: Override in UI**
- Open Settings page
- Enter custom values for endpoints or models
- Leave empty to use env/defaults
- See real-time which source is being used

**Example 4: Use HuggingFace Whisper model**
```bash
# .env
WHISPER_MODEL=openai/whisper-large-v3
```
Or set in UI Settings → Model Configuration → Whisper Model Override

**Example 5: Use custom diarization model**
```bash
# .env
DIARIZATION_MODEL=pyannote/speaker-diarization-2.1
```
Or set in UI Settings → Model Configuration → Diarization Model Override

**Example 6: Mix and match**
```bash
# .env
API_BASE_URL=http://production:5001
WHISPER_MODEL=large
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
LLM_API_BASE_URL=http://local-llm:8000/v1
LLM_MODEL=llama-3-70b
```
Then override just the Whisper model in UI to test a different size.

## Testing Checklist
- ✅ Defaults work without .env or UI settings
- ✅ .env overrides defaults
- ✅ UI settings override .env  
- ✅ Empty string in UI falls back properly
- ✅ LLM config passes correctly to server
- ✅ API client reinitializes when endpoint changes
- ✅ Settings page shows correct source indicators
