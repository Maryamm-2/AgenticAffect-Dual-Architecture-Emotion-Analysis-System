class EvaluatorAgent:
    def __init__(self):
        self.performance_metrics = {}
        self.emotion_map = {
            'sadness': 0,
            'joy': 1,
            'love': 2,
            'anger': 3,
            'fear': 4,
            'surprise': 5
        }

    def map_emotion(self, emotion_str):
        return self.emotion_map.get(emotion_str, -1)  # -1 if unknown

    def evaluate(self, predictions, ground_truth):
        pred_emotions = [self.map_emotion(pred['emotion']) for pred in predictions]
        self.performance_metrics['accuracy'] = self.calculate_accuracy(pred_emotions, ground_truth)
        self.performance_metrics['precision'] = self.calculate_precision(pred_emotions, ground_truth)
        self.performance_metrics['recall'] = self.calculate_recall(pred_emotions, ground_truth)
        self.performance_metrics['f1_score'] = self.calculate_f1_score(pred_emotions, ground_truth)
        return self.performance_metrics

    def calculate_accuracy(self, predictions, ground_truth):
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        return correct / len(ground_truth)

    def calculate_precision(self, predictions, ground_truth):
        true_positive = sum((p == g == 1) for p, g in zip(predictions, ground_truth))
        predicted_positive = sum(p == 1 for p in predictions)
        return true_positive / predicted_positive if predicted_positive > 0 else 0

    def calculate_recall(self, predictions, ground_truth):
        true_positive = sum((p == g == 1) for p, g in zip(predictions, ground_truth))
        actual_positive = sum(g == 1 for g in ground_truth)
        return true_positive / actual_positive if actual_positive > 0 else 0

    def calculate_f1_score(self, predictions, ground_truth):
        precision = self.calculate_precision(predictions, ground_truth)
        recall = self.calculate_recall(predictions, ground_truth)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def provide_recommendations(self):
        recommendations = []
        if self.performance_metrics['accuracy'] < 0.7:
            recommendations.append("Consider improving the preprocessing steps.")
        if self.performance_metrics['f1_score'] < 0.6:
            recommendations.append("Explore different classification models.")
        return recommendations