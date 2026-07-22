from app.pipeline import VoiceSentimentPipeline

pipe = VoiceSentimentPipeline()  # charge les 2 modèles (patiente ~1 min sur CPU)
resultat = pipe.run("tests_audio/exemple.wav")
print(resultat)