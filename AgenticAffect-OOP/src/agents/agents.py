from models.emotion_classifier import EmotionClassifier
import pandas as pd

class ResearcherAgent:
    def analyze(self, df: pd.DataFrame):
        summary = {
            "num_samples": len(df),
            "label_distribution": df['label'].value_counts().to_dict(),
            "sample_texts": df['text'].head(5).tolist()
        }
        return summary

class PreprocessorAgent:
    def preprocess(self, texts):
        # Simple normalization: lowercase and strip
        cleaned = [text.lower().strip() for text in texts]
        return cleaned

class ClassifierAgent:
    def __init__(self):
        self.classifier = EmotionClassifier()
    def classify(self, texts):
        return self.classifier.classify_batch(texts)

class EvaluatorAgent:
    def evaluate(self, true_labels, pred_labels):
        correct = sum(t == p for t, p in zip(true_labels, pred_labels))
        accuracy = correct / len(true_labels)
        return {
            "accuracy": accuracy,
            "total": len(true_labels),
            "correct": correct,
            "details": list(zip(true_labels, pred_labels))
        }
