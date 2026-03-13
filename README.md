# MindMend Guardian v2

Offline voice AI safety guardian for families, running on Faraday-caged Raspberry Pi with crisis detection and 988-safe redirects.

## Technical Summary
- **Hardware**: Raspberry Pi with ESP32 wake-word detection, optimized for <1W always-on operation
- **AI Pipeline**: Whisper STT → fused Triton/CUDA kernel inference → voice synthesis
- **Security**: Rust AES-GCM encryption for journals, air-gapped Faraday cage enclosure
- **Safety**: Real-time crisis detection with automatic 988 lifeline integration
- **Performance**: <300ms response times, solar-ready for remote deployment

## Impact
Provides unbreakable privacy and safety for vulnerable users in crisis situations. Built with Michigan grit to ensure families have reliable AI protection when networks fail or threats emerge.

## Quick Start
```bash
git clone https://github.com/MiMindMendinc/mindmend-guardian.git
cd mindmend-guardian
pip install -r requirements.txt
python main.py
```

## Features
- Voice-activated crisis monitoring
- Offline LLM processing
- Encrypted local storage
- Emergency resource integration

## License
MIT - Built for good, no strings attached.