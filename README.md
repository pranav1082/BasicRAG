# Basic RAG (Retrieval-Augmented Generation) Project

A lightweight Python implementation of a **Retrieval-Augmented Generation (RAG)** pipeline. This project allows you to query your private documents (such as  TXT files) by combining a vector database for semantic search with a Large Language Model (LLM) for generating precise, context-aware answers.

## 🚀 Features

- **Document Ingestion:** Reads and chunks unstructured text data or documents.
- **Vector Embeddings:** Converts document chunks into vector representations.
- **Vector Storage:** Stores and indexes embeddings for fast similarity searches.
- **Contextual Generation:** Fetches relevant document blocks and injects them into an LLM prompt to generate grounded answers.

## 🛠️ Prerequisites

Before getting started, make sure you have the following installed:
- Python 3.12 or higher
- Git

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd your-repository
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Environment Configuration

This project requires environment variables (such as API keys) to run. These variables must be kept secure and **should never be committed to GitHub**.

1. Copy the provided template to create your local `.env` file:
   ```bash
   cp .env.example .env
   ```
2. Open the newly created `.env` file and insert your actual credentials:
   ```text
   OPENAI_API_KEY=your_actual_openai_api_key_here
   # Add any other required keys below
   ```

## 💻 Usage

To run the RAG pipeline and start querying your documents, execute the main script:

```bash
python main.py
```

## 📂 Project Structure

```text
├── data/               # Place your source documents (.txt) here
├── .env                # Local secret variables (IGNORED BY GIT)
├── .env.example        # Template for environment configuration
├── .gitignore          # Rules for files Git should ignore
├── main.py             # Main entry point for the RAG workflow
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```
