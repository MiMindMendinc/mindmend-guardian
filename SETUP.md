# Setup and Installation Guide

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA-capable GPU for enhanced performance

### Basic Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MiMindMendinc/mindmend-guardian.git
   cd mindmend-guardian
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test the installation:**
   ```bash
   python tools/test_tts.py
   ```

## Component-Specific Setup

### MindMend Guardian (Voice Assistant)

The voice guardian requires additional dependencies for full functionality:

```bash
# For text-to-speech
pip install pyttsx3

# For voice activity detection
pip install onnxruntime

# For speech recognition (requires whisper.cpp submodule)
git submodule update --init --recursive
# Follow whisper.cpp build instructions for your platform

# For LLM support (optional)
pip install llama-cpp-python
```

**Models needed:**
- Place `silero_vad.onnx` in `models/` directory
- Place `ggml-tiny.en.bin` (Whisper) in `models/` directory
- (Optional) Place LLM model in `models/` directory

**Run the guardian:**
```bash
python guardian/mindmend_guardian.py
```

### Luna Safety Core

Luna requires additional dependencies for threat detection and alerts:

```bash
pip install flask PyJWT firebase-admin spacy spacytextblob
python -m spacy download en_core_web_sm
```

**Run Luna API:**
```bash
python guardian/luna/luna_safety_core.py
```

**Run Luna tests:**
```bash
python guardian/luna/luna_safety_core.py --test
```

### Perrien Simulator

The simulator is already included in core dependencies (streamlit + plotly).

**Run the simulator:**
```bash
streamlit run perrien-simulator/app.py
```

## Development Setup

### Running Tests

The project includes a basic automated test suite:

```bash
# Run basic validation tests
python tests/test_basic.py

# Test TTS phrases
python tools/test_tts.py

# Test Luna Safety Core
python guardian/luna/luna_safety_core.py --test
```

### Code Quality

Ensure your code follows Python best practices:

```bash
# Check syntax
python -m py_compile guardian/mindmend_guardian.py
python -m py_compile guardian/luna/luna_safety_core.py
python -m py_compile perrien-simulator/app.py
```

## Raspberry Pi Installation

For low-power deployment on Raspberry Pi:

1. Use Raspberry Pi OS (64-bit recommended)
2. Install Python 3.8+: `sudo apt install python3 python3-pip python3-venv`
3. Install audio dependencies: `sudo apt install portaudio19-dev python3-pyaudio`
4. Follow the basic installation steps above
5. Consider using lighter model variants for better performance

## Troubleshooting

### PyAudio Installation Issues

On Linux:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

On macOS:
```bash
brew install portaudio
pip install pyaudio
```

On Windows:
- Download prebuilt wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
- Install with: `pip install PyAudio‑*.whl`

### Torch/CUDA Issues

For CPU-only installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA support, visit: https://pytorch.org/get-started/locally/

### Missing Models

Models are not included in the repository due to size. Download them separately:

- Silero VAD: https://github.com/snakers4/silero-vad
- Whisper: https://github.com/ggerganov/whisper.cpp
- LLM models: https://huggingface.co/

## Privacy & Security

- All processing happens locally on your device
- No internet connectivity required for core features
- No telemetry or data collection
- See `CONTRIBUTING.md` for security guidelines

## Getting Help

- Check the [CONTRIBUTING.md](CONTRIBUTING.md) guide
- Open an issue on GitHub
- Review existing issues for common problems

## Next Steps

Once installed, explore:
- Voice activation with "hey mindmend"
- Perrien Simulator for data analysis
- Luna Safety Core for child safety features

For development, see [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and workflow.
