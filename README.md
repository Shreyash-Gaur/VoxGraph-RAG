
# VoxGraph-RAG: Agentic Hybrid Graph-Vector RAG System

## Project Overview

VoxGraph-RAG is an advanced, fully local Retrieval-Augmented Generation (RAG) system that marries the structural reasoning capabilities of Knowledge Graphs with the semantic search power of Vector Databases—both natively unified within Neo4j. By combining dense vector similarity with structural graph traversal, this hybrid RAG architecture resolves a critical flaw in standard vector search: the loss of relational context between entities. To guarantee high availability and retrieval robustness, the system implements a local FAISS index as a seamless fail-safe mechanism.

Designed for complex enterprise reasoning, the system employs an agentic orchestration layer (LangGraph) to intelligently route queries, grade retrieved documents, and formulate multi-step responses. It features a robust multimodal interface, offering a real-time, local voice agent via LiveKit (Faster-Whisper STT & Piper TTS) alongside a Chainlit web interface, delivering a seamless, low-latency user experience.

## Key Features

* **Hybrid Graph-Vector Retrieval:** Simultaneously queries FAISS (for semantic context) and Neo4j (for multi-hop entity relationships via Cypher), merging the results for comprehensive context grounding.
* **Agentic Orchestration (LangGraph):** Implements an advanced state machine with conditional routing. The agent dynamically routes to conversational or retrieval flows, grades document relevance, and transforms queries if retrieved context is insufficient.
Here is how you can perfectly capture this in your Key Features or Design Decisions section:
* **Dual-Purpose FAISS Architecture:** Strategically leverages FAISS at two distinct layers of the pipeline. At the routing layer, an in-memory FAISS index powers a Semantic Cache to instantly serve exact or semantically identical past queries. At the retrieval layer, a separate disk-backed FAISS index operates as a highly resilient vector fallback mechanism, ensuring 100% context availability even if the primary Neo4j vector store experiences downtime.
* **Production-Grade Reranking:** Integrates a CrossEncoder/BGE Reranker to re-score and re-order the combined graph and vector outputs, drastically reducing context bloat and improving LLM precision.
* **Query Expansion (HyDE):** Utilizes Hypothetical Document Embeddings to expand sparse queries, bridging the lexical gap between user questions and technical documentation.
* **Multi-Tier Caching Architecture:** Features a SQLite-backed Embedding Cache to prevent redundant compute on identical documents, and a FAISS-backed Semantic Cache to instantly serve answers to previously asked, semantically similar queries.
* **Local Voice Agent (LiveKit):** Features a fully local, GPU-accelerated voice pipeline using `faster-whisper` for STT and `piper-tts` for low-latency voice synthesis, totally bypassing cloud dependencies.
* **Tool Calling Capabilities:** Allows the LLM to autonomously trigger external tools, such as a localized sandboxed mathematical calculator, for deterministic problem-solving.

## System Architecture

```text
                                  +-----------------------+
                                  |  User Input (Voice/UI)|
                                  +-----------+-----------+
                                              |
                                      [ Chainlit / LiveKit ]
                                              |
                                     +--------v---------+
                             HIT     |  Semantic Cache  |-----> [ Instant Response ]
                          +----------|  (FAISS + SQLite)|
                          |          +--------+---------+
                          |                 MISS
                          v                   |
               +-------------------+          v
               |  Conversational   |<---[ LangGraph Agent ]---> [ Query Expander (HyDE) ]
               |  Router / Memory  |          |
               +-------------------+          |
                                    +---------v----------+
                                    |  Hybrid Retriever  |
                                    +----+----------+----+
                                         |          |
                      +------------------v--+     +-v------------------+
                      |    Neo4j Graph      |     |  FAISS Vector DB   |
                      | (Entity Traversal)  |     | (Dense Embeddings) |
                      |(Native Vector Index)|     |     (Fallback)     |
                      +------------------+--+     +-+------------------+
                                         |          |
                                    +----v----------v----+
                                    | CrossEncoder/BGE   |
                                    |     Reranker       |
                                    +---------+----------+
                                              |
                                    +---------v----------+
                                    | Ollama LLM / Tools |-----> [ Final Output ]
                                    +--------------------+

```

## Tech Stack

| Category | Technology |
| --- | --- |
| **Orchestration & Agents** | LangChain, LangGraph |
| **LLMs & Embeddings** | Ollama (`phi4-mini`), `mxbai-embed-large:latest` |
| **Vector Database** | FAISS (CPU) |
| **Graph Database** | Neo4j (`langchain-neo4j`) |
| **Reranking** | `FlagEmbedding` (BAAI/bge-reranker-v2-m3), `sentence-transformers` |
| **Backend & APIs** | FastAPI, Pydantic, Uvicorn |
| **Memory & Caching** | SQLite (WAL mode), SentenceTransformers (`BAAI/bge-large-en-v1.5`) |
| **Voice & Frontend** | LiveKit, Chainlit, `faster-whisper`, `piper-tts` |

## Project Structure

```text
├── backend/
│   ├── agents/               # LangGraph state machine and agent definitions
│   ├── core/                 # App configuration, custom exceptions, and logging
│   ├── models/               # Pydantic schemas for request/response validation
│   ├── scripts/              # Async directory watchers and ingestion pipelines
│   ├── services/             # Core logic: Retrieve, Memory, Graph, and Cache services
│   └── tools/                # LLM tools (Calculator, HyDE, Embedder, Reranker)
├── interfaces/
│   ├── voice_mode/           # LiveKit Voice Agent and custom local audio plugins
│   └── web_chat/             # Chainlit UI application
├── docker-compose.yaml       # Infrastructure configuration (Neo4j, LiveKit server)
└── requirements.txt          # Python dependencies

```

## Getting Started

### Prerequisites

* Python 3.10+
* Docker & Docker Compose
* Ollama installed locally

### Installation & Setup

1. **Clone the repository and install dependencies:**
```bash
git clone <repo_url>
cd voxgraph-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```


2. **Pull Required Local Models via Ollama:**
```bash
ollama pull phi4-mini
ollama pull mxbai-embed-large:latest

```


3. **Spin up Infrastructure (Neo4j & LiveKit):**
```bash
docker-compose up -d

```


4.**Turn on the Watcher for document Ingestion**
```bash
python backend/scripts/ingest_vector_watch.py --watch knowledge --interval 10
python backend/scripts/ingest_graph_watch.py --watch knowledge

```


5. **Start the Backend API:**
```bash
uvicorn backend.main:APP --reload --port 8000

```


6. **Launch the User Interface:**
```bash
chainlit run interfaces/web_chat/chainlit_app.py -w -h --port 8001

```



## Configuration

The system is highly configurable via environment variables or the `.env` file (parsed by `pydantic-settings`). Key variables include:

* `OLLAMA_MODEL` (default: `phi4-mini`)
* `EMBEDDING_MODEL` (default: `mxbai-embed-large:latest`)
* `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
* `RERANKER_ENABLED` (default: `True`)
* `USE_HYDE` (default: `True`)
* `SEMANTIC_CACHE_THRESHOLD` (default: `0.80`)

## How It Works — Technical Deep Dive

At the core of VoxGraph-RAG is a resilient **Hybrid Retrieval** engine. When a user submits a complex query, the system first triggers an Entity Extraction chain via an LLM structured output to identify target nodes, constructing a Cypher query against **Neo4j** to retrieve explicit structural relationships up to 2 hops away. Simultaneously, a dense vector search is executed. The system primarily leverages **Neo4j's native vector store** for this semantic search to unify infrastructure; however, it implements a strict **Fail-Safe Mechanism** where if the Neo4j vector index is unreachable or yields no candidates, the system instantly cascades to a local, disk-backed **FAISS** index to guarantee context delivery.

To bridge the lexical gap between short user questions and dense technical documentation, the retrieval phase is augmented with **HyDE (Hypothetical Document Embeddings)**. Before vector retrieval, a low-temperature LLM generates a factual, hypothetical answer which is appended to the search query, pulling mathematically closer nearest-neighbors from the vector space. To manage the massive volume of context generated by this expanded dual retrieval, a **CrossEncoder Reranker** scores every retrieved chunk against the original query, applies a sigmoid normalization to the logits, and aggressively filters out low-relevance noise before context reaches the LLM.

The entire control flow is governed by an intelligent **LangGraph State Machine**. The graph evaluates user intent (routing trivial queries directly to a memory-injected chitchat node), grades the retrieved documents for factual relevance, and automatically triggers a query transformation loop if the context is deemed insufficient (Self-RAG/CRAG methodology). Finally, the generation node features **Agentic Tool Calling**; if the model detects a deterministic mathematical problem within the query, it autonomously suspends text generation, delegates the expression to a sandboxed Python **Calculator Tool** via `numexpr`, and injects the exact computed result back into its reasoning trace to entirely eliminate arithmetic hallucinations.

## Performance & Design Decisions

* **Modular Service Separation:** The system architecture enforces strict separation of concerns by isolating core logic into independent services (e.g., `RetrieveService`, `GraphService`, `SemanticCacheService`, and `MemoryService`). This modularity drastically improves maintainability and ensures seamless future upgrades.
* **Semantic Caching:** To prevent redundant compute and reduce expensive LLM API invocations, a secondary FAISS index (`BAAI/bge-large-en-v1.5`) caches previous query-answer pairs. Semantically similar queries return instant cache hits, bypassing the generation step entirely.
* **Retrieval Robustness via FAISS Fallback:** The hybrid retrieval engine guarantees context delivery by utilizing a graceful fallback mechanism; if the primary Neo4j vector search yields no candidates, the system seamlessly cascades to the local FAISS vector store.
* **Hallucination Mitigation via Cross-Encoder:** Context retrieved from both graph and dense vector sources is aggregated and passed through a CrossEncoder/FlagEmbedding reranker. This normalizes scores and filters out low-relevance noise, tightening the context window and minimizing LLM hallucinations.
* **Decoupled Voice Pipeline:** The LiveKit voice agent (`interfaces/voice_mode`) operates entirely independently from the backend retrieval logic, interacting with the core RAG brain exclusively via RESTful payloads.
* **Asynchronous Processing:** The UI and voice interfaces utilize `aiohttp` and `cl.make_async` to make non-blocking HTTP requests for LLM calls and transcriptions. This ensures the application remains highly responsive and avoids thread-locking during heavy generative workloads.


## Future Improvements

* **Multi-Agent Collaboration:** Transition from a single router agent to a specialized multi-agent framework (e.g., a dedicated Graph Agent, a Vector Agent, and a Math Agent) managed by a central supervisor.
* **Dynamic / Semantic Chunking:** Implement layout-aware parsing and semantic chunking (grouping sentences by embedding similarity) rather than static recursive character splitting to improve boundary context.
* **Scalable Vector Stores:** Migrate the local FAISS index to a scalable, persistent cloud solution like Milvus or Pinecone for distributed enterprise deployments.
* **Evaluation Pipeline:** Integrate Ragas or TruLens to establish automated CI/CD benchmarks for context precision, context recall, and answer faithfulness.

## Author

**Shreyash Gaur** | Gen AI Engineer
[LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/shreyashgaur/) | shreyashgaur01@gmail.com