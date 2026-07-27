"""Évaluation quantitative du pipeline :
- WER (Word Error Rate) pour la transcription ASR ;
- Accuracy et F1-macro pour la classification de sentiment.
"""
import json
import unicodedata

from jiwer import wer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from app.pipeline import VoiceSentimentPipeline


def normaliser(texte: str) -> str:
    """Normalisation avant calcul du WER : minuscules, sans ponctuation ni accents.

    L'ASR sort du texte brut sans ponctuation ; comparer sans normaliser
    pénaliserait injustement des différences de forme, pas de contenu.
    """
    texte = texte.lower()
    texte = "".join(
        c for c in unicodedata.normalize("NFD", texte)
        if unicodedata.category(c) != "Mn"
    )
    texte = "".join(c if c.isalnum() or c.isspace() else " " for c in texte)
    return " ".join(texte.split())


def main():
    with open("evaluation/annotations.json", encoding="utf-8") as f:
        annotations = json.load(f)

    pipe = VoiceSentimentPipeline()

    wers, y_vrai, y_pred = [], [], []

    print(f"{'Fichier':<38} {'WER':>6}  {'Attendu':<8} {'Prédit':<8}")
    print("-" * 68)

    for item in annotations:
        resultat = pipe.run(item["fichier"])

        w = wer(normaliser(item["reference"]), normaliser(resultat["transcription"]))
        wers.append(w)
        y_vrai.append(item["sentiment"])
        y_pred.append(resultat["sentiment"])

        ok = "✅" if resultat["sentiment"] == item["sentiment"] else "❌"
        print(f"{item['fichier']:<38} {w:>5.0%}  {item['sentiment']:<8} "
              f"{resultat['sentiment']:<8} {ok}")

    labels = ["positif", "neutre", "négatif"]
    print("\n===== RÉSULTATS =====")
    print(f"WER moyen (ASR)        : {sum(wers) / len(wers):.1%}")
    print(f"Accuracy (sentiment)   : {accuracy_score(y_vrai, y_pred):.1%}")
    print(f"F1-macro (sentiment)   : {f1_score(y_vrai, y_pred, average='macro'):.3f}")
    print("\nMatrice de confusion (lignes = vérité, colonnes = prédiction)")
    print(f"{'':>10} {labels[0]:>9} {labels[1]:>9} {labels[2]:>9}")
    for label, ligne in zip(labels, confusion_matrix(y_vrai, y_pred, labels=labels)):
        print(f"{label:>10} {ligne[0]:>9} {ligne[1]:>9} {ligne[2]:>9}")


if __name__ == "__main__":
    main()