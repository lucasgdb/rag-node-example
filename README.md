# Retrieval-Augmented Generation (RAG)

A simple RAG pipeline that answers JavaScript questions over your own `.txt` documents, using LangChain’s in-memory vector store and a locally-hosted Ollama LLM.

---

## 🔍 What It Does

1. **Loads** your `docs/javascript.txt` file
2. **Splits** it into “documents” by sentence (splitting on `.`)
3. **Embeds** each chunk with `nomic-embed-text`
4. **Indexes** embeddings in memory (no external vector DB)
5. **Retrieves** the top 2 most-similar chunks for any question
6. **Prompts** your Ollama model with those chunks + strict instructions
7. **Returns** a concise, accurate answer (max 3 sentences) or “Insufficient information…”

---

## ⚙️ Prerequisites

- **Node.js** ≥ 24
- **Ollama** installed & running locally
