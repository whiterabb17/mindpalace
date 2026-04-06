# mem-retriever: HNSW Memory Retriever

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Layer: 6](https://img.shields.io/badge/Layer-6-blue.svg)

`mem-retriever` provides the sixth layer of the **MindPalace** memory pipeline. It offers high-performance semantic search and context reconstruction (RAG) by integrating `RuVector-Core` HNSW indexing with a structured relational fact graph.

## 🗝️ Key Features

- **Integrated Semantic Search**: Seamlessly combines sub-millisecond vector retrieval with relational fact constraints.
- **Dual-Store Strategy**: Supports multiple `VectorStore` implementations:
  - **InMemoryStore**: Volatile, high-speed storage for rapid prototyping.
  - **RuVectorStore**: Production-grade, persistent storage using HNSW indexing and `GraphDB`.
- **Knowledge Hydration**: Automatically rebuilds the semantic vector index from the persistent KnowledgeBase at startup, ensuring zero data loss across session restarts.
- **Graph-Augmented RAG**: Filters semantic results based on relational graph status (e.g., automatically excluding superseded or invalidated facts).
- **Context Bootstrapping**: Synthesizes high-level agent memory states from disparate facts using LLM-driven reconstruction.
- **Dynamic Category Filtering**: Supports scoped search for facts within specific technical or persona-based categories.

## 🏗️ Core Mechanism

The `MemoryRetriever` orchestrates the retrieval flow:
1. **Embedding**: Generates a vector for the user's query.
2. **Search**: Performs an HNSW search in the `VectorStore`.
3. **Filtering**: Cross-references results with the `FactGraph` to ensure veracity and relevance.
4. **Reconstruction**: (Optional) Uses an LLM to synthesize the retrieved facts into a coherent system prompt.

## 🛠️ Usage Example

```rust
use mem_retriever::{MemoryRetriever, RuVectorStore, DistanceMetric};
use mem_core::{FileStorage, EmbeddingProvider, LlmClient, FactGraph};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let storage = FileStorage::new(std::path::PathBuf::from("./data"));
    let embeddings: Arc<dyn EmbeddingProvider> = Arc::new(MyEmbeddingProvider::new());
    let llm: Arc<dyn LlmClient> = Arc::new(MyLlmClient::new());
    let graph = Arc::new(FactGraph::new(None)?);
    
    let store = Arc::new(RuVectorStore::new(1536, DistanceMetric::Cosine, Arc::clone(&graph)));
    let retriever = MemoryRetriever::new(storage, embeddings, llm, store, graph);
    
    // Search for relevant facts before the next agent turn
    let facts = retriever.retrieve_relevant_facts("What is the project goal?", 5, None).await?;
    
    Ok(())
}
```

## 📂 Architecture Context
As **Layer 6**, `mem-retriever` is the primary interface for semantic memory. It runs after fact extraction, providing the agent with a rich, graph-augmented context that maintains long-term coherence across sessions.
