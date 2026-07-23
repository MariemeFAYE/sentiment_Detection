import gradio as gr

from app.pipeline import VoiceSentimentPipeline
from app.audio_utils import AudioError

# Une seule instance : modèles chargés au démarrage, pas à chaque requête
pipeline = VoiceSentimentPipeline()

EMOJIS = {"positif": "😊", "négatif": "😠", "neutre": "😐"}
COULEURS = {"positif": "#22c55e", "négatif": "#ef4444", "neutre": "#94a3b8"}

CSS = """
.resultat-box {
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    font-size: 1.4em;
    font-weight: 600;
}
footer {display: none !important;}
"""


def analyser(audio_path, progress=gr.Progress()):
    if audio_path is None:
        return (
            "<div class='resultat-box'>⚠️ Aucun fichier audio fourni.</div>",
            "",
            None,
        )
    try:
        progress(0.1, desc="Prétraitement de l'audio...")
        progress(0.3, desc="Transcription (Wav2Vec 2.0)...")
        result = pipeline.run(audio_path)
        progress(0.9, desc="Analyse du sentiment (CamemBERT)...")
    except AudioError as e:
        return (
            f"<div class='resultat-box' style='background:#fef2f2;color:#991b1b'>⚠️ {e}</div>",
            "",
            None,
        )

    # Carte de résultat principal
    s = result["sentiment"]
    html = (
        f"<div class='resultat-box' style='background:{COULEURS[s]}22;"
        f"color:{COULEURS[s]}'>{EMOJIS[s]} Sentiment {s.upper()}<br>"
        f"<span style='font-size:0.65em;font-weight:400'>"
        f"Confiance : {result['score']:.0%}</span></div>"
    )

    # Barres de probabilités personnalisées (une par classe)
    barres = "".join(
        f"<div style='margin:6px 0'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.9em'>"
        f"<span>{EMOJIS[label]} {label}</span><span>{score:.0%}</span></div>"
        f"<div style='background:#e2e8f0;border-radius:6px;height:10px'>"
        f"<div style='background:{COULEURS[label]};width:{score:.0%};"
        f"height:100%;border-radius:6px'></div></div></div>"
        for label, score in result["detail"].items()
    )
    detail_html = (
        "<div style='padding:8px 4px'>"
        "<div style='font-size:0.85em;color:#64748b;margin-bottom:8px'>"
        "Probabilités brutes du modèle (avant règle de seuil)</div>"
        f"{barres}</div>"
    )

    return html, result["transcription"], detail_html


with gr.Blocks(title="Sentiment Vocal") as demo:
    gr.Markdown(
        """
        # 🎙️ Détection de sentiment dans les appels vocaux
        Pipeline **Wav2Vec 2.0** (transcription) → **DistilCamemBERT** (sentiment).
        Chargez un appel client en français (.wav / .mp3, max 5 min) ou enregistrez-vous.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio d'entrée",
            )
            btn = gr.Button("🔍 Analyser", variant="primary", size="lg")
            gr.Examples(
                examples=[
                    ["tests_audio/demo_positif.wav"],
                    ["tests_audio/demo_negatif.wav"],
                    ["tests_audio/demo_neutre.wav"],
                    ["tests_audio/exemple.wav"],
                ],
                inputs=[audio_in],
                label="Exemples (positif / négatif / neutre / cas difficile)",
            )
        with gr.Column(scale=1):
            resultat_out = gr.HTML(label="Résultat")
            transcription_out = gr.Textbox(
                label="📝 Transcription intermédiaire (ASR)",
                lines=4,
                interactive=False,
            )
            detail_out = gr.HTML(label="Détail des scores")

    btn.click(
        fn=analyser,
        inputs=[audio_in],
        outputs=[resultat_out, transcription_out, detail_out],
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
    )