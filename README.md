# README: Agentic AI-Powered Document Retrieval and Question Answering System

## Overview
This project sets up an **agentic AI-powered document retrieval and question-answering system** using **LangChain, LangGraph, ChromaDB, and LMStudio**. It consists of three main components:
1. **Web Scraper (`scraper.py`)** – Scrapes web pages, extracts text, and saves content for embedding and retrieval.
2. **Embedding Generator (`embedder.py`)** – Processes text documents, generates embeddings, and stores them in ChromaDB.
3. **Retrieval & Question Answering Agent (`agent_base.py`)** – Uses a language model (Hermes 3) to retrieve, rank, and generate responses based on a structured LangGraph workflow.

## Features
- **Agentic Decision-Making**: Uses **LangGraph** to structure decision-making with nodes.
- **Document Embedding**: Converts text documents into embeddings using `HuggingFaceBgeEmbeddings`.
- **Vector Database Storage**: Stores document embeddings in `ChromaDB` for fast retrieval.
- **LLM-powered QA System**: Uses `Hermes-3-Llama-3.2-3B` via LMStudio for answering user queries.
- **Multi-step Retrieval & Ranking**: Retrieves relevant documents, generates multiple responses, and ranks them based on similarity.
- **Web Scraping**: Extracts web content and integrates it into the embedding database for enhanced retrieval.

---
## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- LMStudio (Running locally on `http://127.0.0.1:1234`)
- Required Python libraries:

```sh
pip install langchain chromadb langgraph numpy scikit-learn requests beautifulsoup4
```

---
## Usage
### 1. Scraping Web Content (`scraper.py`)
This script scrapes a website, extracts text, and saves it as a text file for embedding.

#### Running the script:
```sh
python scraper.py
```

#### How it works:
- Fetches web pages and extracts text.
- Saves cleaned text in `./content/`.
- Supports recursive scraping up to a specified depth.

---
### 2. Generating Document Embeddings (`embedder.py`)
This script loads text documents from `./new_datasets/microlabs_usa/`, generates embeddings, and stores them in `./content/chroma_db/`.

#### Running the script:
```sh
python embedder.py
```

#### How it works:
- Loads `.json` files from the specified dataset directory.
- Splits text into chunks of 500 characters with 50-character overlap.
- Uses **BAAI/bge-base-en** to generate embeddings.
- Stores embeddings in ChromaDB.

---
### 3. Running the Question-Answering Agent (`agent_base.py`)
This script initializes an **agentic retrieval-augmented LLM** that:
1. Uses **LangGraph** to determine whether to retrieve documents or generate a response directly.
2. Retrieves relevant documents from ChromaDB.
3. Generates multiple response variations using **qa_chain**.
4. Evaluates similarity between generated responses and retrieved documents.
5. Ranks responses and selects the best one.

#### Running the script:
```sh
python agent_base.py
```

#### How it works:
- Initializes **Hermes-3-Llama-3.2-3B** (via LMStudio) as the LLM.
- Loads document embeddings from ChromaDB for retrieval.
- Uses **LangGraph** to manage retrieval and response generation through nodes.
- Uses cosine similarity to rank responses.
- Outputs the **best-ranked answer** with a confidence score.

##### Example Query:
```python
query = "What are the contraindications of Acetazolamide?"
output = agent_workflow.invoke({"query": query})

print("Ranked Response: ", output["ranked_response"])
print("Confidence Score: ", output["confidence_score"])
```

---
## Configuration
You can modify the following settings:
- **Dataset Directory** (`embedder.py`): `data_directory = "./new_datasets/microlabs_usa"`
- **ChromaDB Storage** (`embedder.py` & `agent_base.py`): `embedding_directory = "./content/chroma_db"`
- **Scraper Output Directory** (`scraper.py`): `data_directory = "./content"`
- **LLM Model Name** (`agent_base.py`): `model_name="hermes-3-llama-3.2-3b"`
- **Temperature for Response Variability** (`agent_base.py`): `temperature=0.7`

---
## Notes
- Make sure LMStudio is **running** before launching `agent_base.py`.
- You can adjust the chunk size and overlap in `embedder.py` for better document splitting.
- Modify the **LangGraph workflow** in `agent_base.py` to refine decision logic.
- Scraper can be modified to limit depth or filter domains.

---
## Future Improvements
- Implement **GPT-4 or Mixtral** for better accuracy.
- Improve ranking using **Reinforcement Learning (RLHF)**.
- Enhance web scraping capabilities with **proxy support**.
- Integrate **custom fine-tuned LLMs** for domain-specific QA.

---
## Author
Developed by [Your Name].

For issues and contributions, open a PR or create an issue in the repository.

---
✅ **Enjoy using your AI-powered retrieval system! 🚀**

