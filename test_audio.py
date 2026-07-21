from app.audio_utils import load_and_preprocess
import os
os.makedirs("tests_audio", exist_ok=True)

wav = load_and_preprocess("tests_audio/exemple.wav")
print(wav.shape, wav.dtype, wav.abs().max())  # torch.Size([N]) float32 1.0