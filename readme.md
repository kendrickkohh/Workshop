# Workshop on Security on Large Language Models

## Installation

- All relevant installations are in requirements.txt
- We also reccomend utilizing venv

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# or
venv\Scripts\activate      # Windows
```

- To install all dependencies, refer to `requirements.txt`

```bash
pip install -r requirements.txt
```

## Demo Components

### get_embedding.py

Handles embedding generation using the `text-embedding-ada-002` embedding model.

### guardrail.py

Implements [LLM Guard](https://github.com/protectai/llm-guard) to sanitize and validate both inputs and outputs from the LLM.

### populate_database.py

Populates the vector datastore by:

- Checking the `Upload_documents/` folder for new entries
- Adding any new data into the datastore if not already present

### query_data.py

Main querying interface that:

- Accepts user queries and sends them to the LLM
- Uses RAG (Retrieval-Augmented Generation) to retrieve relevant context from the datastore
- Applies input/output validation using `guardrail.py`

### Ollama models

- Llama3.2
- mxbai-embed-large

## Acknowledgements

- Created by: Koh Yihao Kendrick
- Mentored by: Professor Ong Chin Ann
