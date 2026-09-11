# Adaptive RAG Chat API

A self-hosted, fully local **Adaptive RAG** pipeline: a LangGraph agent that routes each question to either a local vector store or a live web search, grades its own answer for hallucination and usefulness, retries when needed, and remembers a rolling summary of the conversation. Everything runs against local OpenAI-compatible model servers — no calls to a hosted LLM API.

## Why this exists

Most RAG demos either always hit the vector store or always hit the web. This project adds two things on top of a standard RAG-Fusion retriever:

- **Query routing** — an LLM decides per-question whether the vector store (Python documentation, in this setup) or a live Google search is the better source.
- **Self-correction** — each generation is graded for whether it's actually grounded in the retrieved documents and whether it answers the question, with automatic retries before the answer is returned.
- **Short-term memory** — a rolling summary of the conversation is persisted between calls and expires automatically after 24 hours, so the API stays stateless per-request while still supporting follow-up questions.

## Architecture

![System architecture overview](https://github.com/user-attachments/assets/744dfe2f-e944-4214-90b5-9262f78d57c0)

| Piece | File | Role |
|---|---|---|
| API | `flask.py` | Exposes `POST /square`, the single HTTP entry point |
| Orchestration | `mainscript.py` | Builds and runs the LangGraph "Adaptive RAG" state machine |
| Retrieval | `fetch.py` | RAG-Fusion: multi-query generation → Chroma similarity search → reciprocal rank fusion → SQLite lookup |
| Ingestion | `docloader.py` | One-off script that parses PDFs into headered chunks and populates Chroma + SQLite |
| Memory | `conv_memory.py` | Stores/retrieves a single rolling conversation summary in its own Chroma collection |

### Document ingestion

`docloader.py` walks a local folder of PDFs, converts each to markdown, and splits on markdown headers. For every chunk it stores two things:

- the **header path** (e.g. `Intro-Setup-Installation`), embedded and written to a Chroma collection — this is what gets searched
- the **chunk text itself**, written to a SQLite table keyed by the same id — this is what gets returned as context

![Document ingestion pipeline](https://github.com/user-attachments/assets/0050e7c7-f32a-4e13-be55-af35354053ca)

### Query flow

`mainscript.py` compiles a LangGraph graph with conditional edges for routing, retries, and memory:

1. **initialize** — load the saved conversation summary (if any, and if not expired)
2. **route_if_relevant** — is the new question related to that summary? If yes, rewrite it as a standalone question first
3. **route_question** — an LLM classifies the question as `web_search` or `vectorstore`
4. **retrieve** / **web_search** — fetch context from Chroma+SQLite or from Google Search
5. **generate** — answer the question from the retrieved context
6. **grade_generation** — check the answer is grounded in the documents and actually answers the question; loop back to `generate` or `web_search` on failure (capped at one retry), otherwise continue
7. **summarize_conversation** — fold the exchange into the rolling summary and persist it

![Adaptive RAG LangGraph flow](https://github.com/user-attachments/assets/dddc0a7e-d7bd-4b66-b80a-8bba76fbe0e5)

### Retrieval detail (RAG-Fusion)

`fetch.py` doesn't do a single similarity search — it generates four related search queries, runs each against Chroma, and merges the ranked results with reciprocal rank fusion before pulling the winning chunks' full text out of SQLite.

![RAG-Fusion retrieval detail](https://github.com/user-attachments/assets/cf82bf94-64bc-4ee0-8104-2bef67e1c773)

## Quick start

### 1. Prerequisites

- Python 3.10+
- Two local OpenAI-compatible model servers reachable on the ports below (e.g. `llama.cpp`'s `server`, LM Studio, or similar):
  - `http://localhost:8081/v1` serving a chat model (the code uses `gemma-2-2b-it.Q6_K.gguf`)
  - `http://localhost:8082/v1` serving an embedding model (`mxbai-embed-large-v1-f16`)
- A Google Programmable Search Engine (CSE) ID and API key, for the web-search fallback

### 2. Install dependencies

```bash
pip install langchain langchain-openai langchain-chroma langchain-community \
            langchain-google-community langgraph flask flask-restful \
            pdf4llm
```

### 3. Set configuration

Edit `docloader.py` to point `folder_path` at your PDFs, and set your Google credentials as environment variables (or directly in `mainscript.py`, where they're currently read):

```bash
export GOOGLE_CSE_ID="your-cse-id"
export GOOGLE_API_KEY="your-api-key"
```

### 4. Ingest documents (one-off)

```bash
python docloader.py
```

This populates `chroma_db/` (header-path embeddings) and `my_database.db` (full chunk text) from the PDFs in your data folder.

### 5. Run the API

```bash
python flask.py
```

### 6. Ask a question

```bash
curl -X POST http://127.0.0.1:5000/square \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I define a decorator in Python?"}'
```

Response:

```json
{ "answer": "..." }
```

## Configuration reference

| Setting | Where | Default |
|---|---|---|
| Chat LLM endpoint | `mainscript.py`, `fetch.py` | `http://localhost:8081/v1` |
| Embedding endpoint | `docloader.py`, `fetch.py`, `conv_memory.py` | `http://localhost:8082/v1` |
| Vector store path | `docloader.py`, `fetch.py` | `chroma_db/` |
| Memory store path | `conv_memory.py` | `conv_memory_db/` |
| SQLite database | `docloader.py`, `fetch.py` | `my_database.db` |
| PDF source folder | `docloader.py` | `folder_path` (hardcoded, update per machine) |
| Conversation memory TTL | `conv_memory.py` | 24 hours |
| Retrieval breadth | `fetch.py` | 4 generated queries × top-3 Chroma matches |
| Hallucination/answer retry cap | `mainscript.py` | 1 retry |


RAG Pipline
![Screenshot 2025-04-19 152510](https://github.com/user-attachments/assets/fed8b4b0-d972-4fd6-900e-dddd0004c477)

Document Parsing
![Screenshot 2025-04-19 153007](https://github.com/user-attachments/assets/1170799d-8bc0-41df-a199-bc89d38190b3)

Retrival Node
![Screenshot 2025-04-19 152715](https://github.com/user-attachments/assets/38b14b2a-d63e-4f91-b4a0-974cfb896190)

Conversation memory logic
![Screenshot 2025-04-19 152449](https://github.com/user-attachments/assets/150e7398-3e02-47d2-b482-45500aac5863)
