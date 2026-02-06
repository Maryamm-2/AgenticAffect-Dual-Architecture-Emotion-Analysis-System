from models.emotion_classifier import EmotionClassifier

class ClassifierAgent:
    def __init__(self):
        self.classifier = EmotionClassifier()

    def classify(self, texts):
        results = self.classifier.classify_batch(texts)
        predictions = []
        for text, (emotion, confidence) in zip(texts, results):
            predictions.append({
                "text": text,
                "emotion": emotion,
                "confidence": confidence
            })
        return predictions