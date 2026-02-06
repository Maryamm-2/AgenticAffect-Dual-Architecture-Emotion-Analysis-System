from transformers import pipeline

class EmotionClassifier:
    def __init__(self):
        self.model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

    def classify(self, text):
        predictions = self.model(text)
        return [(pred['label'], pred['score']) for pred in predictions]

    def classify_batch(self, texts):
        predictions = self.model(texts)
        return [(pred['label'], pred['score']) for pred in predictions]