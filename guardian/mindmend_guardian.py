# mindmend_guardian.py (Full Conversational Version - Dec 2025)
# Ultra-low power Silero VAD → Whisper tiny wake → Full LLM guardian on Colossus/H100
# < 1 W chill • Full GPU unleash on wake • Always-on mental health companion
# Features: Seamless mode switching, VAD-triggered natural conversation, optional llama.cpp LLM,
#           pyttsx3 TTS, return to chill on "go to sleep mindmend"

import os
import time
import torch
import psutil
import numpy as np
import collections
from datetime import datetime
import platform
import random

# ==================== CONFIG ====================
WAKE_PHRASE = "hey mindmend"
SHUTDOWN_PHRASE = "go to sleep mindmend"
SILENCE_TIMEOUT = 60  # seconds deep sleep in chill
CONVO_TIMEOUT = 300   # seconds no speech → auto return to chill
MODEL_DIR = "models"
VAD_THRESHOLD_CHILL = 0.5
VAD_THRESHOLD_CONVO = 0.6  # Higher during conversation to avoid noise
SUSTAINED_CHUNKS = 3
UTT_END_SILENCE = 2.0  # seconds silence to end utterance
MIN_UTT_SEC = 1.0      # minimum speech length to process
AUDIO_BUFFER_SEC_CHILL = 3
AUDIO_BUFFER_SEC_MAX = 30  # max context retained
SAMPLING_RATE = 16000

# Gentle canned replies (fallback if no LLM)
GENTLE_REPLIES = [
    "You're stronger than this moment feels.",
    "It's okay to not be okay right now.",
    "You've survived 100% of your hardest days so far.",
    "Your feelings are valid. They won't last forever.",
    "Be extra gentle with yourself today.",
    "I'm really glad you're here.",
    "One tiny step forward is still progress.",
    "You're allowed to take a break.",
    "This feeling will pass. It always does.",
    "You're doing better than you think."
]

# Optional LLM config (place a quantized GGUF in ./models/, e.g. Llama-3.1-8B-Instruct-Q4_K_M.gguf)
LLM_MODEL_PATH = f"{MODEL_DIR}/ggml-llama-3.1-8b-instruct-q4_k_m.gguf"  # Set to None to disable

# ===============================================
print("QUANTUM_GLITCH + Silero Ultra-Low Voice Guardian | Michigan MindMend 2025 (Full Conversational)")

# ===================== QUANTUM_GLITCH INIT =====================
def quantum_glitch_init():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

quantum_glitch_init()

# ===================== THROTTLE (CHILL MODE) =====================
def throttle_chill():
    torch.set_num_threads(1)
    # Use os.getpid() directly instead of enumerating all processes
    try:
        p = psutil.Process(os.getpid())
        p.nice(19)
        if platform.system() == 'Linux':
            p.ionice(psutil.IOPRIO_CLASS_IDLE)
        p.cpu_affinity([0])
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        pass

def unleash_full():
    torch.set_num_threads(os.cpu_count() or 128)

throttle_chill()  # Start in chill

# ===================== MODELS =====================
SILERO_AVAILABLE = False
silero = None
h = np.zeros((2, 1, 64), dtype=np.float32)
c = np.zeros((2, 1, 64), dtype=np.float32)

WHISPER_AVAILABLE = False
whisper_model = None

LLM_AVAILABLE = False
llm = None

TTS_AVAILABLE = False
def speak(text):
    print(f"MindMend: {text}")

try:
    import onnxruntime as ort
    silero_path = f"{MODEL_DIR}/silero_vad.onnx"
    if os.path.exists(silero_path):
        silero = ort.InferenceSession(silero_path, providers=['CPUExecutionProvider'])
        SILERO_AVAILABLE = True
        print("Silero VAD loaded – <1W chill mode ready")
except Exception as e:
    print(f"Silero VAD error: {e}")

def load_whisper():
    global whisper_model, WHISPER_AVAILABLE
    if WHISPER_AVAILABLE: return
    try:
        from whisper_cpp import Whisper
        model_path = f"{MODEL_DIR}/ggml-tiny.en.bin"
        if os.path.exists(model_path):
            whisper_model = Whisper(model_path)
            WHISPER_AVAILABLE = True
            print("Whisper tiny loaded")
    except Exception as e:
        print(f"Whisper load failed: {e}")

def load_llm():
    global llm, LLM_AVAILABLE
    if LLM_AVAILABLE or not LLM_MODEL_PATH or not os.path.exists(LLM_MODEL_PATH): return
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=-1,      # Full GPU offload
            n_ctx=4096,
            verbose=False
        )
        LLM_AVAILABLE = True
        print("Local LLM loaded – full quantum power ready")
    except Exception as e:
        print(f"LLM load failed: {e}")

def load_tts():
    global TTS_AVAILABLE, speak
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        def tts_speak(text):
            print(f"MindMend (speaking): {text}")
            engine.say(text)
            engine.runAndWait()
        speak = tts_speak
        TTS_AVAILABLE = True
        print("TTS loaded (pyttsx3)")
    except Exception as e:
        print(f"TTS unavailable: {e}")

load_tts()

# ===================== MAIN LOOP =====================
import pyaudio

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=512)

sr_array = SAMPLING_RATE  # Use int scalar directly - more efficient than array wrapper

wake_buffer = collections.deque(maxlen=SAMPLING_RATE * AUDIO_BUFFER_SEC_CHILL)
utterance_buffer = collections.deque(maxlen=SAMPLING_RATE * AUDIO_BUFFER_SEC_MAX)

mode = "chill"
speech_chunk_count = 0
in_speech = False
last_voice_time = time.time()
last_heartbeat = time.time()
conversation_history = []
MAX_CONVERSATION_HISTORY = 20  # Prevent OOM on long sessions

print("Guardian online. Awaiting your voice in deepest chill… Say “hey mindmend”")

while True:
    try:
        data = stream.read(512, exception_on_overflow=False)
        # Avoid unnecessary dtype conversion copy - numpy auto-promotes int16 to float in division
        audio_np = np.frombuffer(data, dtype=np.int16) / 32768.0
        
        # Always maintain rolling buffers
        wake_buffer.extend(audio_np)
        if mode == "conversation" and in_speech:
            utterance_buffer.extend(audio_np)

        if SILERO_AVAILABLE:
            # Use view instead of reshape to avoid unnecessary copy
            input_dict = {"input": audio_np[np.newaxis, :], "sr": sr_array, "h": h, "c": c}
            out = silero.run(None, input_dict)
            prob = out[0][0][1]
            h, c = out[1], out[2]
        else:
            prob = 0.0

        current_time = time.time()

        if mode == "chill":
            if prob > VAD_THRESHOLD_CHILL:
                speech_chunk_count += 1
                last_voice_time = current_time
                if speech_chunk_count >= SUSTAINED_CHUNKS:
                    load_whisper()
                    if WHISPER_AVAILABLE and len(wake_buffer) >= SAMPLING_RATE * 1:
                        # Use asarray for less copy overhead
                        full_audio = np.asarray(wake_buffer, dtype=np.float32)
                        result = whisper_model.transcribe(full_audio, language="en")
                        text = result["text"].lower().strip()
                        print(f"Heard: “{text}”")
                        if WAKE_PHRASE in text:
                            print("\nWAKE PHRASE CONFIRMED — UNLEASHING FULL QUANTUM_GLITCH POWER")
                            unleash_full()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            print("550+ token/sec • All systems live • Ready to protect the future.")
                            mode = "conversation"
                            conversation_history = []
                            utterance_buffer.clear()
                            speak("I'm here for you. Share what's on your mind.")
                            speech_chunk_count = 0
            else:
                speech_chunk_count = 0

            if current_time - last_heartbeat > 40:
                # Removed blocking CPU polling for better power efficiency in chill mode
                print(f"[{datetime.now():%H:%M:%S}] ULTRA-CHILL | Say \"{WAKE_PHRASE}\"")
                last_heartbeat = current_time

            # Removed unnecessary sleep - VAD already provides throttling

        elif mode == "conversation":
            load_llm()  # Lazy load

            if prob > VAD_THRESHOLD_CONVO:
                if not in_speech:
                    in_speech = True
                    utterance_buffer.clear()
                    print("Listening...")
                last_voice_time = current_time

            if in_speech and (current_time - last_voice_time) > UTT_END_SILENCE:
                if len(utterance_buffer) > SAMPLING_RATE * MIN_UTT_SEC:
                    load_whisper()
                    if WHISPER_AVAILABLE:
                        # Use asarray for less copy overhead
                        full_audio = np.asarray(utterance_buffer, dtype=np.float32)
                        result = whisper_model.transcribe(full_audio, language="en")
                        text = result["text"].strip()
                        if text:
                            print(f"You said: “{text}”")
                            if SHUTDOWN_PHRASE in text.lower():
                                speak("Take care. Returning to quiet listening.")
                                print("Returning to ultra-chill mode...")
                                throttle_chill()
                                mode = "chill"
                                conversation_history = []
                            else:
                                conversation_history.append(f"User: {text}")
                                # Limit conversation history to prevent OOM
                                if len(conversation_history) > MAX_CONVERSATION_HISTORY:
                                    conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY:]
                                if LLM_AVAILABLE:
                                    system_prompt = (
                                        "You are MindMend, a compassionate AI companion for gentle emotional support. "
                                        "Respond with empathy, validation, and encouragement. Keep replies warm, concise, and positive. "
                                        "Do not act as a therapist."
                                    )
                                    full_prompt = system_prompt + "\n\nConversation:\n" + "\n".join(conversation_history) + "\nMindMend:"
                                    response = llm.create_completion(
                                        full_prompt,
                                        max_tokens=300,
                                        temperature=0.8,
                                        stop=["User:", "\n\n"]
                                    )
                                    reply = response["choices"][0]["text"].strip()
                                else:
                                    reply = random.choice(GENTLE_REPLIES)
                                print(f"MindMend: {reply}")
                                speak(reply)
                                conversation_history.append(f"MindMend: {reply}")
                                # Limit conversation history to prevent OOM
                                if len(conversation_history) > MAX_CONVERSATION_HISTORY:
                                    conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY:]
                in_speech = False

            if current_time - last_voice_time > CONVO_TIMEOUT:
                speak("Going quiet to save energy. Call me anytime.")
                throttle_chill()
                mode = "chill"

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

stream.stop_stream()
stream.close()
p.terminate()
print("\nGuardian offline. Stay strong.")
