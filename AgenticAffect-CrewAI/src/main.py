# CrewAI with Ollama: Emotion Analysis Pipeline
# This demonstrates CrewAI agents using Ollama for open-source LLMs.
# Assumes Ollama is installed and running locally with a model (e.g., llama3.2).

from datasets import load_dataset
from crewai import Agent, Crew, Task, Process
from crewai.llm import LLM
from crewai.tools import BaseTool
from transformers import pipeline
import pandas as pd

# Step 1: Load Dataset
print("Loading emotion dataset...")
dataset = load_dataset("emotion")
train = dataset["train"].to_pandas().sample(10, random_state=42)
texts = train["text"].tolist()
labels = train["label"].tolist()

# Step 2: Set up LLM with Ollama (Open-source, local)
# Agentic AI: LLMs enable reasoning and communication.
llm = LLM(
    model="ollama/llama3.2",  # Correct format for Ollama
    base_url="http://localhost:11434"  # Ollama's default port
)

# Step 3: Define Emotion Classifier Tool
class EmotionClassifierTool(BaseTool):
    name: str = "Emotion Classifier"
    description: str = "Classifies emotions in texts using a Hugging Face model."

    def __init__(self):
        super().__init__()
        self._model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

    def _run(self, texts: str) -> str:
        text_list = [t.strip() for t in texts.split(',')]
        predictions = self._model(text_list)
        results = [{"text": text, "emotion": pred["label"], "confidence": pred["score"]}
                   for text, pred in zip(text_list, predictions)]
        return str(results)

classifier_tool = EmotionClassifierTool()

# Step 4: Define Agents
researcher = Agent(
    role="Dataset Researcher",
    goal="Analyze dataset statistics.",
    backstory="Expert in NLP datasets.",
    llm=llm,
    verbose=True
)

preprocessor = Agent(
    role="Text Preprocessor",
    goal="Clean and prepare text.",
    backstory="Specializes in text preprocessing.",
    llm=llm,
    verbose=True
)

classifier = Agent(
    role="Emotion Classifier",
    goal="Predict emotions using tools.",
    backstory="Expert in emotion detection.",
    llm=llm,
    verbose=True,
    tools=[classifier_tool]
)

evaluator = Agent(
    role="Evaluator",
    goal="Assess performance.",
    backstory="Evaluates AI outputs.",
    llm=llm,
    verbose=True
)

# Step 5: Define Tasks
task1 = Task(
    description=f"Analyze dataset with {len(train)} samples.",
    expected_output="Dataset summary.",
    agent=researcher
)

task2 = Task(
    description=f"Preprocess texts: {texts[:5]}.",
    expected_output="Preprocessed texts.",
    agent=preprocessor
)

task3 = Task(
    description=f"Classify emotions: {', '.join(texts)}.",
    expected_output="Predictions.",
    agent=classifier
)

task4 = Task(
    description="Evaluate results.",
    expected_output="Metrics.",
    agent=evaluator
)

# Step 6: Create and Run Crew
crew = Crew(
    agents=[researcher, preprocessor, classifier, evaluator],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
    verbose=True
)

print("\nStarting CrewAI with Ollama...")
result = crew.kickoff()
print("\n===== FINAL RESULT =====")
print(result)