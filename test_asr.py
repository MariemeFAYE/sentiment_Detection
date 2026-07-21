from app.audio_utils import load_and_preprocess
from app.asr import ASRModel

asr = ASRModel()  # 1er lancement : téléchargement ~1,2 Go, sois patiente
wav = load_and_preprocess("tests_audio/exemple.wav")
print("Transcription :", asr.transcribe(wav))