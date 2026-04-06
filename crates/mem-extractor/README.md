# mem-extractor: Fact Extraction & Knowledge Distillation

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Layer: 5](https://img.shields.io/badge/Layer-5-blue.svg)

`mem-extractor` provides the fifth layer of the **MindPalace** memory pipeline. It focuses on identifying and distilling durable, high-confidence facts from ongoing conversations and committing them to a persistent, relational knowledge graph.

## 🗝️ Key Features

- **Durable Fact Distillation**: Uses LLMs to analyze context and extract structured JSON facts, including categories, confidence scores, and cross-fact dependencies.
- **Semantic Deduplication**: Leverages vector embeddings and cosine similarity to prevent redundant knowledge entry, keeping the knowledge base lean.
- **Conflict Resolution**: Integrated `ConflictResolver` service that uses LLM arbitration to detect and resolve contradictions within the knowledge base.
- **Relational Graph Commitment**: Facts are stored in a structured graph using `ruvector-graph`, with support for versioning and links for superseded information.
- **Event-Driven Extraction**: Includes a `ReflectionLayer` that triggers fact extraction based on specific user cues (e.g., "actually," "remember that").
- **Multi-Agent Scoping**: Supports `Private`, `Project`, and `Global` fact visibility for fleet-wide learning.

## 🏗️ Core Mechanism

The `FactExtractor` uses a multi-pass approach to manage fact integrity:
1. **Extraction**: LLM generates a set of potential facts from the context history.
2. **Deduplication**: Semantic similarity check against the existing knowledge base.
3. **Commitment**: Atomic update to the persistent graph.
4. **Resolution**: Post-commitment scan for category-level contradictions and LLM-driven truth arbitration.

## 🛠️ Usage Example

```rust
use mem_extractor::FactExtractor;
use mem_core::{Context, LlmClient, EmbeddingProvider, FileStorage, MindPalaceConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm: Arc<dyn LlmClient> = Arc::new(MyLlmClient::new());
    let embeddings: Arc<dyn EmbeddingProvider> = Arc::new(MyEmbeddingProvider::new());
    let storage = FileStorage::new(std::path::PathBuf::from("./data"));
    let config = MindPalaceConfig::default();
    
    let extractor = FactExtractor::new(
        llm,
        embeddings,
        storage,
        config,
        "knowledge.json".into(),
        "session-123".into()
    );
    
    let mut context = Context::default();
    
    // Extract and commit durable facts during the optimization cycle
    extractor.process(&mut context).await?;
    
    Ok(())
}
```

## 📂 Architecture Context
As **Layer 5**, `mem-extractor` transforms volatile conversational history into permanent knowledge. It runs after filtering and summarization, ensuring that the facts extracted are from a high-quality, processed context.
