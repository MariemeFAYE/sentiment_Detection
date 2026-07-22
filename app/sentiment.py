import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cmarkea/distilcamembert-base-sentiment"

# Le modèle sort 5 classes (indices 0..4 = 1 à 5 étoiles).
# On les regroupe en 3 classes conformément au sujet.
LABEL_GROUPS = {
    "négatif": [0, 1],   # 1 et 2 étoiles
    "neutre":  [2],      # 3 étoiles
    "positif": [3, 4],   # 4 et 5 étoiles
}


class SentimentModel:
    """Classification de sentiment en 3 classes avec DistilCamemBERT."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        """Retourne {"sentiment": ..., "score": ...} pour un texte français."""
        if not text or not text.strip():
            raise ValueError("Texte vide : impossible d'analyser le sentiment.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,      # coupe à 512 tokens (limite de BERT)
            max_length=512,
        ).to(self.device)

        logits = self.model(**inputs).logits          # forme (1, 5)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # probabilités qui somment à 1

        # Regroupement 5 classes -> 3 classes : on somme les probabilités
        scores = {
            label: probs[indices].sum().item()
            for label, indices in LABEL_GROUPS.items()
        }
        sentiment = max(scores, key=scores.get)

        # Règle de décision : si aucune classe ne domine clairement,
        # on considère le message comme neutre (les vrais sentiments
        # tranchés produisent des scores > 0.9 en pratique).
        CONFIDENCE_THRESHOLD = 0.5
        if scores[sentiment] < CONFIDENCE_THRESHOLD:
            sentiment = "neutre"

        return {
            "sentiment": sentiment,
            "score": round(scores[sentiment], 4),  # score de confiance demandé par le sujet
            "detail": {k: round(v, 4) for k, v in scores.items()},
        }