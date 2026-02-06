from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

class PreprocessorAgent:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def tokenize(self, text):
        # Tokenize the cleaned text
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        # Remove stopwords from the tokenized text
        return [word for word in tokens if word not in self.stop_words]

    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        tokens = self.remove_stopwords(tokens)
        return tokens

    def preprocess_batch(self, texts):
        return [self.preprocess(text) for text in texts]