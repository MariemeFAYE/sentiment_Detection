from app.pipeline import VoiceSentimentPipeline

pipe = VoiceSentimentPipeline()

for nom, attendu in [
    ("demo_positif", "positif"),
    ("demo_negatif", "négatif"),
    ("demo_neutre", "neutre"),
]:
    r = pipe.run(f"tests_audio/{nom}.wav")
    verdict = "✅" if r["sentiment"] == attendu else "❌"
    print(f"{verdict} {nom}: {r['sentiment']} ({r['score']:.0%})")
    print(f"   Transcription : {r['transcription']}")