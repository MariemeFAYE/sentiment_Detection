import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.audio_utils import AudioError
from app.pipeline import VoiceSentimentPipeline

EXTENSIONS_ACCEPTEES = {".wav", ".mp3"}
pipeline: VoiceSentimentPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles UNE FOIS au démarrage du serveur."""
    global pipeline
    pipeline = VoiceSentimentPipeline()
    yield  # l'API tourne ; rien à nettoyer à l'arrêt


app = FastAPI(
    title="API Sentiment Vocal",
    description=(
        "Analyse le sentiment d'un appel vocal en français.\n\n"
        "**Pipeline** : audio → prétraitement (16 kHz mono) → "
        "transcription (Wav2Vec 2.0) → sentiment (DistilCamemBERT, 3 classes).\n\n"
        "Formats acceptés : `.wav`, `.mp3` — durée max : 5 minutes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    """Réponse du pipeline d'analyse."""
    transcription: str = Field(
        ..., description="Texte transcrit par le modèle ASR",
        examples=["je suis très satisfaite de votre service"],
    )
    sentiment: str = Field(
        ..., description="Classe prédite : positif, négatif ou neutre",
        examples=["positif"],
    )
    score: float = Field(
        ..., description="Score de confiance de la classe prédite (0 à 1)",
        examples=[0.9824],
    )
    detail: dict[str, float] = Field(
        ..., description="Probabilité de chacune des 3 classes",
        examples=[{"négatif": 0.0038, "neutre": 0.0138, "positif": 0.9824}],
    )


@app.get("/", tags=["Statut"], summary="Vérifier que l'API est en ligne")
def health():
    return {"statut": "ok", "modeles_charges": pipeline is not None}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Analyse"],
    summary="Analyser le sentiment d'un fichier audio",
    responses={
        400: {"description": "Fichier invalide (format, vide, silencieux, trop long)"},
        500: {"description": "Erreur interne du pipeline"},
    },
)
async def predict(
    file: UploadFile = File(..., description="Fichier audio .wav ou .mp3 (max 5 min)"),
):
    """Transcrit l'audio puis classe le sentiment en 3 classes.

    Retourne la transcription intermédiaire, le sentiment,
    le score de confiance et le détail des probabilités.
    """
    # 1. Validation de l'extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in EXTENSIONS_ACCEPTEES:
        raise HTTPException(
            status_code=400,
            detail=f"Extension '{ext}' non supportée. Formats acceptés : .wav, .mp3",
        )

    # 2. Sauvegarde dans un fichier temporaire (le pipeline lit un chemin)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 3. Analyse, avec conversion des erreurs métier en réponses HTTP propres
    try:
        return pipeline.run(tmp_path)
    except AudioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)  # nettoyage du fichier temporaire dans tous les cas