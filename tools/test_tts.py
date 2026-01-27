# Simple test script for TTS/alert phrases.
# This script is intentionally minimal and uses standard library only when possible.
# It will print phrases to stdout; if "pyttsx3" is installed, it will speak them.

PHRASES = [
    "Hey, are you okay? I noticed something—please check your surroundings.",
    "This area is off-limits. Please return to a safe place.",
    "Warning: Unexpected approach detected nearby. Stay with a trusted adult.",
]

try:
    import pyttsx3
    engine = pyttsx3.init()
    for p in PHRASES:
        print(p)
        engine.say(p)
    engine.runAndWait()
except Exception:
    # Fallback: just print
    for p in PHRASES:
        print(p)

print("\nDone. To hear audio, install 'pyttsx3' and run this script again.")
