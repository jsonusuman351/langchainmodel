
# 🚀 LLM & Semantic Search Playground

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python) ![LangChain](https://img.shields.io/badge/LangChain-0086CB?style=for-the-badge&logo=langchain) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai) ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD61E?style=for-the-badge&logo=huggingface) ![Gemini](https://img.shields.io/badge/Gemini-8E75B8?style=for-the-badge&logo=google)

Welcome to the LLM & Semantic Search Playground! This repository is a hands-on guide and collection of scripts demonstrating how to interact with various **Large Language Models (LLMs)** and **Embedding Models**. It showcases the foundational concepts behind modern AI applications like RAG (Retrieval-Augmented Generation).

This project explores:
-   Invoking different LLMs (**Open-Source** vs. **Closed-Source**).
-   Using models via **APIs** (like OpenAI, Gemini) vs. running them **locally**.
-   Generating **text embeddings** to capture semantic meaning.
-   Performing **semantic search** using cosine similarity to find the most relevant documents.

---
---

### 📸 Demos & Examples

Here are some screenshots showing the key scripts in action.

**1. OpenAI Chat Model (`openaichatmodel.py`)**
*Shows a simple conversation with the GPT model via API.*
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/ca2123b2-378f-4879-a912-33f7f58364dd" />

**2. Hugging Face Chat Model (`huggingface_chatmodel_local.py`)**
*Demonstrates running an open-source model locally on your machine.*
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/6aba7f24-7cfe-4099-9535-30626199c511" />

**3. Generating Embeddings (`embedding_openai_docs.py`)**
*Displays the numerical vector representations (embeddings) of text documents.*
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/7237ecea-c989-49f2-a153-e4c06f5c8394" />

**4. Semantic Search (`document_similarity.py`)**
*Shows the script taking a query and finding the most relevant document using cosine similarity.*
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/493731c8-ddf1-4963-9230-de433512784d" />

---


### ✨ Core Concepts Demonstrated

This repository is structured around three key concepts of modern AI development:

1.  **🤖 LLM & Chat Model Invocation**:
    -   **Closed-Source (API-based)**: Scripts to interact with powerful models like OpenAI's GPT and Google's Gemini through their APIs.
    -   **Open-Source (Local & API)**: Examples of running open-source models from platforms like Hugging Face and DeepSeek, both by downloading them locally and using their APIs.

2.  **🔍 Text Embedding Generation**:
    -   Demonstrates how to convert text (documents and user queries) into numerical vectors (embeddings) using models like OpenAI's `text-embedding-ada-002` and local Hugging Face models. These embeddings capture the *meaning* of the text, not just the keywords.

3.  **💡 Semantic Search**:
    -   A practical mini-project (`document_similarity.py`) that shows how embeddings are used. It converts a user's query into a vector and uses **cosine similarity** to find the most contextually relevant document from a knowledge base. This is the core mechanism behind vector databases and RAG applications.

---

### 🛠️ Tech Stack

-   **Core Framework**: LangChain
-   **LLM/Chat Model Providers**: OpenAI, Google (Gemini), Hugging Face, DeepSeek
-   **Embedding Models**: OpenAI, Sentence-Transformers (from Hugging Face)
-   **Core Libraries**: `langchain`, `python-dotenv`, `numpy`, `scikit-learn`
-   **Local Model Execution**: `transformers`, `torch`

---

### ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jsonusuman351/langchainmodel.git](https://github.com/jsonusuman351/langchainmodel.git)
    cd langchainmodel
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # It is recommended to use Python 3.10 or higher
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    To use API-based models (OpenAI, Gemini, etc.), you need to provide your API keys.
    -   Create a file named `.env` in the root directory of the project.
    -   Add your keys to this file like so:
        ```env
        OPENAI_API_KEY="your-openai-api-key"
        GOOGLE_API_KEY="your-google-api-key"
        HF_TOKEN="your-huggingface-api-key"
        DEEPSEEK_API_KEY="your-deepseek-api-key"
        ```

---

### 🚀 Usage Guide

This repository is a collection of standalone scripts. You can run each one to explore a specific concept.

#### 1. Invoking LLMs and Chat Models

These scripts show how to get responses from different models.

-   **OpenAI Chat Model (API):**
    ```bash
    python chatModels/openaichatmodel.py
    ```
-   **Google Gemini Chat Model (API):**
    ```bash
    python chatModels/gemini_chatmodel.py
    ```
-   **Hugging Face Chat Model (Local Download):**
    *Note: This will download the model (~1.5 GB) the first time you run it.*
    ```bash
    python chatModels/huggingface_chatmodel_local.py
    ```
-   **DeepSeek Chat Model (API):**
    ```bash
    python chatModels/deepseekchatmodel.py
    ```

#### 2. Generating Text Embeddings

These scripts demonstrate how to convert text into vector embeddings.

-   **Using OpenAI's API to embed documents:**
    ```bash
    python EmbeddedModels/embedding_openai_docs.py
    ```
-   **Using a local Hugging Face model to embed text:**
    *Note: This will download the embedding model the first time you run it.*
    ```bash
    python EmbeddedModels/embedding_hf_local.py
    ```

#### 3. Performing Semantic Search (Mini-Project)

This script is a complete example of using embeddings for semantic search. It takes a user query, finds the most similar document from a list, and returns it.

-   **Run the semantic search demo:**
    ```bash
    python document_similarity.py
    ```
    **Example Interaction:**
    ```
    (D:\Projects\langchainmodel\venv) D:\Projects\langchainmodel>python document_similarity.py
    tell me about narendra modi
    Narendra Modi is the Prime Minister of India known for his charismatic leadership and economic reforms.
    similarity score is: 0.6063302711097277
    ```

---

### 📂 Code Playground Structure

<details>
<summary>Click to view the folder structure</summary>

```
langchainmodel/
│
├── LLMs/                     # Scripts for basic LLMs
│   └── llm_demo.py
│
├── chatModels/               # Scripts for various chat models
│   ├── openaichatmodel.py    # (OpenAI API)
│   ├── gemini_chatmodel.py     # (Google Gemini API)
│   ├── huggingface_chatmodel_local.py # (Local open-source model)
│   └── ...
│
├── EmbeddedModels/           # Scripts for text embedding models
│   ├── embedding_openai_docs.py # (Using OpenAI API)
│   └── embedding_hf_local.py  # (Using local open-source model)
│
├── document_similarity.py    # Mini-project for semantic search
├── requirements.txt
├── .env                      # ( create this for API keys)
└── README.md
```
</details>

---
````