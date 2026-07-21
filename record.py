import sounddevice as sd
import soundfile as sf

SR = 16_000
DURATION = 10  # secondes

print("Parle maintenant...")
audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
sd.wait()
sf.write("tests_audio/exemple.wav", audio, SR)
print("Enregistré dans tests_audio/exemple.wav")