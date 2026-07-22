from app.sentiment import SentimentModel

sent = SentimentModel()  # télécharge ~270 Mo au 1er lancement (léger !)

for texte in [
    "je suis très satisfaite de votre service merci beaucoup",
    "c'est inadmissible je n'ai toujours pas reçu ma commande je suis furieuse",
    "je vous appelle pour savoir si le produit est disponible",
]:
    print(texte, "->", sent.predict(texte))