import librosa
import numpy as np
import torch
import torchaudio

TARGET_SR = 16_000
MAX_DURATION_S = 300  # 5 minutes max (spec du sujet)
SILENCE_THRESHOLD = 1e-4

class AudioError(Exception):
    """Erreur métier levée pour un fichier audio invalide."""
    pass

def load_and_preprocess(path: str) -> torch.Tensor:
    """Charge un fichier .wav/.mp3 et retourne un tenseur mono 16 kHz normalisé.

    Étapes : chargement -> mono -> rééchantillonnage 16 kHz -> normalisation.
    Lève AudioError si le fichier est invalide, vide, trop long ou silencieux.
    """
    # 1. Chargement via librosa (gère .wav et .mp3, portable sur Windows)
    #    sr=None : on garde le taux d'échantillonnage d'origine
    #    mono=False : on gère nous-mêmes la conversion mono (pédagogique + contrôlé)
    try:
        audio, sr = librosa.load(path, sr=None, mono=False)
    except Exception as e:
        raise AudioError(f"Format non supporté ou fichier corrompu : {e}")

    # Conversion numpy -> tenseur PyTorch de forme (canaux, échantillons)
    waveform = torch.from_numpy(np.atleast_2d(audio)).float()

    # 2. Fichier vide
    if waveform.numel() == 0:
        raise AudioError("Le fichier audio est vide.")

    # 3. Durée maximale (5 min)
    duration = waveform.shape[1] / sr
    if duration > MAX_DURATION_S:
        raise AudioError(f"Audio trop long ({duration:.0f}s > {MAX_DURATION_S}s).")

    # 4. Conversion en mono (moyenne des canaux)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 5. Rééchantillonnage à 16 kHz (requis par Wav2Vec 2.0)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, int(sr), TARGET_SR)

    # 6. Détection d'audio silencieux
    if waveform.abs().max() < SILENCE_THRESHOLD:
        raise AudioError("Audio silencieux : aucune parole détectable.")

    # 7. Normalisation de l'amplitude dans [-1, 1]
    waveform = waveform / waveform.abs().max()

    return waveform.squeeze(0)  # tenseur 1D attendu par Wav2Vec 2.0