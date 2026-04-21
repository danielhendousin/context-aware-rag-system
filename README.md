# Context-Aware RAG System

An advanced Retrieval-Augmented Generation (RAG) application for chatting with uploaded PDF documents, built as a thesis-driven project around Turkish public companies' financial reports.

## Overview

This project implements a **history-aware, multi-document RAG pipeline** with a Streamlit interface. Users can upload one or more PDF files, ask questions about their contents, and receive source-grounded answers supported by retrieved document chunks.

The repository packages the thesis implementation into a clean, GitHub-ready application structure.

## Key Features

- Multi-PDF upload and ingestion
- Context-aware conversational retrieval
- Follow-up question reformulation using chat history
- FAISS vector store for local similarity search
- MMR retrieval for relevance and diversity
- Source-aware answers with page references
- Streamlit interface for interactive testing
- Support for multiple LLM providers:
  - OpenAI
  - Hugging Face Hub
  - Ollama

## Thesis Context

This repository is based on the undergraduate thesis:

**"Türkiye'deki Kamu Şirketlerinin Finansal Raporlarının LLM Modeli"**  
Sivas Cumhuriyet University, July 2025.

According to the thesis abstract and evaluation chapter, the system was built with **LangChain** and **RAG**, indexed uploaded PDF documents with **FAISS**, and was tested on more than **30 questions**, with 7 representative financial questions analyzed in detail. The thesis reports that the selected examples matched the source PDF content accurately, including numbers, percentages, note references, and table-based values.


## Project Structure

```text
context-aware-rag-system/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── assets/
└── docs/
    ├── thesis.pdf
    └── thesis_summary.md
```

## How It Works

1. Users upload one or more PDF files.
2. The system reads documents with `PyPDFLoader`.
3. Text is split into chunks with `RecursiveCharacterTextSplitter`.
4. Chunks are embedded using a Hugging Face embedding model.
5. Embeddings are stored in a local **FAISS** vector index.
6. A **history-aware retriever** reformulates follow-up questions when needed.
7. Retrieved chunks are passed into a QA chain.
8. The system returns a concise answer and shows source snippets.

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Embeddings
- OpenAI / Hugging Face Hub / Ollama
- PyPDF
- python-dotenv

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create your environment file

Copy `.env.example` to `.env` and add your keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key_here
```

### 3. Run the app

```bash
streamlit run app.py
```

## Suggested Questions

Try questions like:

- `Bana şirket hakkında kısa bir özet ver.`
- `2024 yılında brüt kar ne kadar?`
- `Nakit ve nakit benzerleri kaç TL?`
- `Dipnot 22'ye göre dönem içinde ayrılan karşılıklar ne kadar?`
- `Enflasyon oranı nedir?`

Then ask a follow-up question such as:

- `Peki bu değer 2023 ile karşılaştırıldığında nasıl?`

## Security

Do **not** commit your real `.env` file or any API keys.  
This repository intentionally includes only `.env.example`.

## Future Improvements

- Better table-aware PDF extraction
- Structured output parsing for numeric fields
- Multi-user session support
- Faster caching and persistent vector index reuse
- Optional FastAPI backend + separate frontend
- Better citation highlighting in the UI

## Author
**Danyal Hendousinabad**
