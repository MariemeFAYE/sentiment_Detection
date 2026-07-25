# Image de base légère avec Python 3.11
FROM python:3.11-slim

# Dépendances système pour le traitement audio (mp3, wav)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Installer les dépendances d'abord (couche mise en cache par Docker :
#    tant que requirements.txt ne change pas, cette étape n'est pas refaite)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 2. Copier le code du projet
COPY app/ ./app/
COPY api/ ./api/
COPY demo/ ./demo/
COPY tests_audio/ ./tests_audio/

# Port de l'API
EXPOSE 8000

# Lancement de l'API (0.0.0.0 obligatoire pour être accessible hors du conteneur)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]