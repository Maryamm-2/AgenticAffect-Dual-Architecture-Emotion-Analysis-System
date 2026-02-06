# Architecture Comparison: OOP vs. CrewAI

This document provides a detailed comparison between the **Object-Oriented Programming (OOP)** implementation and the **CrewAI (Agentic)** implementation of the emotion analysis pipeline.

| Feature | AgenticAffect-OOP | AgenticAffect-CrewAI |
| :--- | :--- | :--- |
| **Core Philosophy** | **Imperative**: You explicitly define *how* everything happens step-by-step. | **Declarative/Agentic**: You define *what* should happen (goals) and let Agents decide *how*. |
| **Control Flow** | Rigid, linear execution in `Pipeline.run()`. Steps are hardcoded: A -> B -> C. | Dynamic execution managed by `Crew`. Agents can theoretically react, retry, or use tools autonomously. |
| **Logic Implementation**| Logic is written in Python methods (e.g., `clean_text` has regex replacement). | Logic is driven by LLM Prompts (e.g., "You are an expert preprocessor. Clean this text."). |
| **Tool Usage** | Direct function calls (`classifier.classify()`). | Tool abstraction (`EmotionClassifierTool`). The Agent "decides" to call the tool. |
| **Extensibility** | Requires modifying code (e.g., changing the loop in `pipeline.py`). | Requires adding a new Agent/Task description. The Crew keeps running. |
| **Dependency** | High reliance on specific Python libraries (NLTK, PyTorch) directly in the flow. | High reliance on an LLM (Ollama/Llama) to drive the process + Tools. |

## 1. Deep Dive: The "Researcher" Difference

### OOP Version (`src/agents/researcher.py`)
*   **How it works**: A Python class with a method `summarize_statistics()`.
*   **Logic**: It explicitly runs `df['label'].value_counts()`. It produces *exact*, deterministic numbers.
*   **Pros**: Fast, 100% accurate math.
*   **Cons**: Can only calculate exactly what you programmed it to calculate.

### CrewAI Version (`src/main.py` -> Researcher Agent)
*   **How it works**: An LLM Agent with the role "Dataset Researcher".
*   **Logic**: The LLM reads the dataset description or sample and generates a summary in natural language.
*   **Pros**: Can provide qualitative insights (e.g., "This dataset looks unbalanced...").
*   **Cons**: Slower; might hallucinate numbers if not given a calculation tool.

## 2. Deep Dive: The "Classifier" Difference

### OOP Version (`src/agents/classifier.py`)
*   **How it works**: A wrapper around the Hugging Face model.
*   **Logic**: Input Text -> DistilBERT -> Output Labels.
*   **Flow**: The `Pipeline` loop forces every text through this function.

### CrewAI Version (`src/main.py` -> Classifier Agent)
*   **How it works**: An Agent equipped with `EmotionClassifierTool`.
*   **Logic**: The Agent receives a task "Classify these texts". It "thinks": *"I need to classify. I have a tool for valid classification. I will call EmotionClassifierTool with input X."*
*   **Flow**: The Agent invokes the tool, receives the JSON result, and then parses it into a final answer.

## 3. Which one to choose?

*   **Choose OOP (AgenticAffect-OOP) if:**
    *   You need speed and efficiency.
    *   You require deterministic, 100% reproducible results.
    *   The process is a fixed pipeline (ETL, batch processing).
    
*   **Choose CrewAI (AgenticAffect-CrewAI) if:**
    *   You want the system to handle ambiguity (e.g., "Analyze the weirdest comments").
    *   You want the agents to collaborate and chat about the results.
    *   You need flexibility (e.g., the Preprocessor might decide *not* to remove stopwords for certain sentences).
