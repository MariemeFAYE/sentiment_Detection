import os
import sys

import sounddevice as sd
import soundfile as sf

SR = 16_000
DURATION = 10  # secondes

nom = sys.argv[1] if len(sys.argv) > 1 else "exemple"
os.makedirs("tests_audio", exist_ok=True)
chemin = f"tests_audio/{nom}.wav"

print(f"Parle maintenant ({DURATION} s)...")
audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
sd.wait()
sf.write(chemin, audio, SR)
print(f"Enregistré dans {chemin}")