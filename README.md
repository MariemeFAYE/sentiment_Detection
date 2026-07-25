# 🎙️ Détection de Sentiment dans les Appels Vocaux

> Projet Final — Module Deep Learning 2 — Dakar Institute of Technology (2026)

Pipeline automatisé qui transcrit un appel vocal client en français (**Wav2Vec 2.0**) puis classe le sentiment exprimé en trois classes — **positif**, **négatif**, **neutre** — avec un score de confiance (**DistilCamemBERT**).

---

## 1. Architecture

```
Audio (.wav / .mp3)
      │
      ▼
┌─────────────────────┐
│   Prétraitement     │  conversion mono, rééchantillonnage 16 kHz,
│  (audio_utils.py)   │  normalisation d'amplitude, validations
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   ASR — Wav2Vec 2.0 │  transcription français → texte
│      (asr.py)       │  découpage en segments de 30 s (mémoire CPU)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Sentiment — BERT    │  DistilCamemBERT 5 classes → regroupement
│   (sentiment.py)    │  en 3 classes + règle de seuil de confiance
└─────────────────────┘
      │
      ▼
{ transcription, sentiment, score, detail }
```

Le pipeline est exposé de deux manières :
- une **interface Gradio** (`demo/gradio_app.py`) pour la démonstration interactive ;
- une **API REST FastAPI** (`api/main.py`) pour l'intégration dans d'autres applications.

## 2. Modèles utilisés et justification

| Rôle | Modèle | Justification |
|---|---|---|
| ASR | [jonatasgrosman/wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french) | Wav2Vec 2.0 fine-tuné sur le français (Common Voice) ; modèle recommandé par le sujet ; bon compromis qualité/ressources sur CPU |
| Sentiment | [cmarkea/distilcamembert-base-sentiment](https://huggingface.co/cmarkea/distilcamembert-base-sentiment) | CamemBERT = BERT pré-entraîné sur le français, cohérent avec des appels francophones ; version distillée ≈ 2× plus rapide (important sur CPU) ; fine-tuné sur des avis clients (Amazon/Allociné), domaine proche des appels clients |

### Choix d'ingénierie

**Regroupement 5 → 3 classes.** Le modèle de sentiment prédit 5 classes (1 à 5 étoiles). Conformément au sujet, elles sont regroupées en sommant les probabilités : 1–2 ⭐ → *négatif*, 3 ⭐ → *neutre*, 4–5 ⭐ → *positif*. La somme des trois scores vaut toujours 1, ce qui donne au score de confiance une interprétation probabiliste propre.

**Règle de seuil pour le neutre.** Les messages réellement positifs ou négatifs produisent des scores très tranchés (> 90 % observés en pratique). Lorsqu'aucune classe ne dépasse **50 %** de confiance, le message est classé *neutre* : l'hésitation du modèle est interprétée comme une absence d'émotion dominante. Le score affiché reste la probabilité brute de la classe retenue — un score bas signale donc honnêtement une prédiction incertaine.

## 3. Installation

Prérequis : Python ≥ 3.9, ~4 Go d'espace disque (modèles inclus).

```bash
git clone https://github.com/MariemeFAYE/sentiment_Detection.git
cd sentiment_Detection

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

Au premier lancement, les modèles (~1,5 Go) sont téléchargés automatiquement depuis Hugging Face et mis en cache.

## 4. Utilisation

### Interface Gradio

```bash
python -m demo.gradio_app
```

Puis ouvrir http://127.0.0.1:7860 — chargement d'un fichier **ou** enregistrement direct au micro, affichage de la transcription intermédiaire, du sentiment avec code couleur et des probabilités par classe.

### API REST

```bash
uvicorn api.main:app
```

Documentation interactive (Swagger) : http://127.0.0.1:8000/docs

**Exemple d'appel curl :**

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@tests_audio/demo_positif.wav"
```

**Exemple d'appel Python :**

```python
import requests

with open("tests_audio/demo_positif.wav", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/predict", files={"file": f})
print(r.status_code, r.json())
```

**Réponse type (200) :**

```json
{
  "transcription": "je suis très satisfaite de votre service tout était parfait merci beaucoup",
  "sentiment": "positif",
  "score": 0.96,
  "detail": {"négatif": 0.01, "neutre": 0.03, "positif": 0.96}
}
```

**Gestion des erreurs (400) :** format non supporté, fichier vide, audio silencieux ou durée > 5 minutes renvoient un code 400 avec un message explicite, par exemple :

```json
{"detail": "Extension '.txt' non supportée. Formats acceptés : .wav, .mp3"}
```

## 5. Structure du projet

```
sentiment_Detection/
├── app/
│   ├── audio_utils.py      # prétraitement + validations (AudioError)
│   ├── asr.py              # transcription Wav2Vec 2.0
│   ├── sentiment.py        # classification DistilCamemBERT (3 classes + seuil)
│   └── pipeline.py         # orchestration de bout en bout
├── api/main.py             # API REST FastAPI (POST /predict)
├── demo/gradio_app.py      # interface Gradio
├── tests_audio/            # fichiers audio de démonstration (un par classe)
├── record.py               # utilitaire d'enregistrement micro
├── test_*.py               # scripts de test de chaque module
└── requirements.txt
```

## 6. Démonstration

Trois fichiers de test sont fournis dans `tests_audio/`, un par classe :

| Fichier | Sentiment attendu | Sentiment prédit | Confiance |
|---|---|---|---|
| `demo_positif.wav` | positif | ✅ positif | 96 % |
| `demo_negatif.wav` | négatif | ✅ négatif | 100 % |
| `demo_neutre.wav` | neutre | ✅ neutre | 26 %* |

\* Score bas attendu : la classe *neutre* est décidée par la règle de seuil lorsque le modèle n'exprime aucune conviction — le score reflète cette incertitude (voir § 2).

Vérification reproductible :

```bash
python test_demos.py
```

## 7. Limites connues

- **Noms propres** : le modèle ASR décode caractère par caractère sans modèle de langage ; les noms propres absents de ses données d'entraînement sont approximés phonétiquement et peuvent dégrader les mots voisins (observé : « je m'appelle Marième Faye » → « au chez ma palmarien feis apples »).
- **Sortie ASR brute** : transcription en minuscules, sans ponctuation — comportement normal du décodage CTC greedy.
- **Messages mitigés** : un avis mi-positif mi-négatif (« le produit est bien mais la livraison trop lente ») produit des scores partagés et est classé *neutre* par la règle de seuil — choix assumé mais discutable selon le cas d'usage.
- **Robustesse du sentiment à l'ASR imparfait** : même avec une transcription partiellement erronée, les mots porteurs d'émotion suffisent souvent au classifieur (observé : transcription dégradée classée positive à 96 %). L'inverse reste possible si un mot-clé émotionnel est mal transcrit.
- **Seuil de confiance (0.5)** : hyperparamètre fixé empiriquement ; il pourrait être optimisé sur un jeu de données annoté.
- **Langue** : pipeline conçu pour le français uniquement.
- **Performance CPU** : compter quelques dizaines de secondes de traitement par minute d'audio sur un CPU standard.

## 8. Auteure

**Marième FAYE** — Master 2 Intelligence Artificielle et Data Science, Dakar Institute of Technology.