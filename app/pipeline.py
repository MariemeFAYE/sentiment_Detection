from app.audio_utils import load_and_preprocess, AudioError
from app.asr import ASRModel
from app.sentiment import SentimentModel


class VoiceSentimentPipeline:
    """Pipeline complet : fichier audio -> transcription -> sentiment."""

    def __init__(self):
        # Les deux modèles sont chargés une seule fois ici
        self.asr = ASRModel()
        self.sentiment = SentimentModel()

    def run(self, audio_path: str) -> dict:
        """Analyse un fichier audio et retourne le résultat complet.

        Lève AudioError pour un fichier invalide (gérée par l'API/Gradio).
        """
        # 1. Prétraitement (mono, 16 kHz, normalisation + validations)
        waveform = load_and_preprocess(audio_path)

        # 2. Transcription
        transcription = self.asr.transcribe(waveform)
        if not transcription.strip():
            raise AudioError("Aucune parole détectée dans l'audio.")

        # 3. Analyse de sentiment
        result = self.sentiment.predict(transcription)

        # 4. Réponse au format demandé par le sujet
        return {
            "transcription": transcription,
            "sentiment": result["sentiment"],
            "score": result["score"],
            "detail": result["detail"],
        }