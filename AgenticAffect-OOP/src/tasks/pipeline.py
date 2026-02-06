from agents.researcher import ResearcherAgent
from agents.preprocessor import PreprocessorAgent
from agents.classifier import ClassifierAgent
from agents.evaluator import EvaluatorAgent
from models.emotion_classifier import EmotionClassifier


class Pipeline:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.researcher = ResearcherAgent()
        self.preprocessor = PreprocessorAgent()
        self.classifier = ClassifierAgent()
        self.evaluator = EvaluatorAgent()

    def run(self):
        # Step 1: Researcher analyzes the raw texts
        print("\n=== Researcher Analysis ===")
        # For demo, just print sample texts and count
        print({
            "num_samples": len(self.texts),
            "sample_texts": self.texts[:5]
        })

        # Step 2: Preprocessor cleans texts
        print("\n=== Preprocessed Texts ===")
        cleaned = self.preprocessor.preprocess_batch(self.texts)
        print(cleaned)

        # Step 3: Classifier predicts emotions
        print("\n=== Classifier Predictions ===")
        predictions = self.classifier.classify(self.texts)
        print(predictions)

        # Step 4: Evaluator assesses results
        print("\n=== Evaluation ===")
        metrics = self.evaluator.evaluate(predictions, self.labels)
        print(metrics)