# AI Assistant

This project implements an AI assistant capable of generating and scoring responses using various natural language processing techniques. The main components include adapting data, handling chat interactions, retrieval-augmented generation (RAG), and scoring responses based on their relevance and accuracy.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Files](#files)
- [Dependencies](#dependencies)

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Quints497/generative-assistants.git
    cd generative-assistants
    ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Setting Up the Environment

Ensure you have a `.env` file in the project directory with the necessary environment variables:

- Add these path to your model in the `.env` file:
  - `YI_MODEL_PATH`
  - `MIXTRAL_MODEL_PATH`
  - `MISTRAL_MODEL_PATH`
  - `NOUS_MODEL_PATH`

### Example Usage

Here is an example of how you can set up and use the different components of the project:

1. **Load Environment Variables:**

    ```python
    from dotenv import load_dotenv
    import os

    load_dotenv("/path/to/your/.env")
    yi_model_path = os.getenv("YI_MODEL_PATH")
    mixtral_model_path = os.getenv("MIXTRAL_MODEL_PATH")
    mistral_model_path = os.getenv("MISTRAL_MODEL_PATH")
    nous_model_path = os.getenv("NOUS_MODEL_PATH")
    ```

2. **Initialise the Adapter and Assistant:**

    ```python
    from adapters.adapter import Adapter
    from assistants.assistant import Assistant
    from assistants.rag_assistant import RagAssistant

    nous_adapter = Adapter(
        model_path=nous_model_path, 
        n_gpu_layers=-1, 
        n_batch=512, 
        n_ctx=4096, 
        verbose=False,
    )
    
    nous_assistant = Assistant(
        adapter=nous_adapter,
        max_tokens=2048,
        temperature=0.5,
        top_k=5,
        top_p=0.1,
        stream=True,
    )
    ```

3. **Start a interactive chat with GUI**

    ```python
    nous_assistant.gradio_chat()
    ```

## Project Structure

```raw
generative-assistants/
├── src/
│   ├── __init__.py
│   ├── adapters/
│   │   └── adapter.py
│   ├── assistants/
│   │   ├── assistant.py
│   │   └── rag_assistant.py
│   └── helpers/
│       └── scoring_responses.py
├── venv/
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## Files

### adapter.py

This script defines the `Adapter` class that interfaces with the Llama model for generating responses based on provided prompts. It includes attributes like `llm` (an instance of the Llama model), `prompt_template` (for formatting prompts), and `generating` (a flag indicating if response generation is in progress).

### assistant.py

This script implements an AI assistant for chat interactions using the `Adapter` for response generation. It maintains a chat history and logs interactions for analysis. Key attributes include `adapter` (for generating responses) and `parameters` (configuration for response generation such as max tokens, temperature, and more).

### rag_assistant.py

This script defines the `RagAssistant` class, which integrates a Retrieval-Augmented Generation (RAG) model. It utilises various modules such as `FlagReranker`, `CharacterTextSplitter`, and `HuggingFaceEmbeddings` to enhance the retrieval and response generation process.

### scoring_responses.py

This script implements a grading algorithm based on cosine similarity. It prepares text representations of both the expected (ideal) responses and the actual responses, then calculates the similarity between them. This is useful for evaluating the quality of generated responses.

## Dependencies

- `llama-cpp-python`
- `python-dotenv`
- `gradio`
- `FlagEmbedding` *
- `langchain_core` *
- `langchain_community` *
- `chromadb` *

`*` Required by rag assistant only
