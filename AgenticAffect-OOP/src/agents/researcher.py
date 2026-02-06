from datasets import load_dataset
import pandas as pd

class ResearcherAgent:
    def __init__(self):
        self.dataset = None

    def load_dataset(self, dataset_name):
        dataset = load_dataset(dataset_name)
        self.dataset = pd.DataFrame(dataset['train'])
        return self.dataset

    def summarize_statistics(self):
        if self.dataset is not None:
            summary = {
                "num_samples": len(self.dataset),
                "label_distribution": self.dataset['label'].value_counts().to_dict(),
                "sample_texts": self.dataset['text'].sample(5).tolist()
            }
            return summary
        else:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

    def analyze_characteristics(self):
        if self.dataset is not None:
            characteristics = {
                "text_length": self.dataset['text'].apply(len).describe(),
                "unique_labels": self.dataset['label'].nunique()
            }
            return characteristics
        else:
            raise ValueError("Dataset not loaded. Please load the dataset first.")