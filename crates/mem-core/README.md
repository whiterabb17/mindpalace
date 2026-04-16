# mem-core: Foundation & Shared Abstractions

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Foundation](https://img.shields.io/badge/Status-Foundational-blue.svg)

`mem-core` is the foundational crate for the **MindPalace** ecosystem. It defines the core data structures, shared traits, and relational modeling primitives used across all memory layers.

## 🗝️ Key Features

- **Relational Knowledge Graph**: Built on top of `ruvector-graph`, providing `FactNode` and `FactGraph` for structured memory.
- **Standard Search Engines**: Includes a high-performance **SqliteSearchEngine** with FTS5 virtual tables for keyword and semantic lookups.
- **Privacy Filtering**: Built-in regex scrubbing for content within `<private>...</private>` tags before fact extraction.
- **Hierarchical Scoping**: Categorize knowledge as `Private` (session), `Project` (team), or `Global` (world-wide technical facts).
- **Short-Term Context**: Robust `Context` and `MemoryItem` structures for managing conversation history.
- **Foundational Traits**:
  - `MemoryLayer`: Standard interface for all context transformation logic.
  - `StorageBackend`: Abstract persistent storage (File, S3, Redis).
  - `ModelProvider`: Generic bridging for LLM (Ollama, Anthropic, etc.) and Embedding providers.
- **Security-First Persistence**: Includes `EncryptedStorageBackend` with transparent **AES-256-GCM** encryption.
- **Observability**: Built-in Prometheus metrics for tracking system health and performance.

## 🏗️ Core Structures

### `FactNode`
A discrete unit of knowledge with confidence scores, versioning, and semantic embeddings.
```rust
pub struct FactNode {
    pub id: String,
    pub content: String,
    pub confidence: f32,
    pub scope: FactScope, // Private, Project, Global
    pub timestamp: u64,
}
```

### `Context`
The agent's active workspace, consisting of a sequence of `MemoryItem` objects.
```rust
pub struct Context {
    pub items: Vec<MemoryItem>,
}
```

## 🛠️ Usage Example: Encrypted Storage

```rust
use mem_core::{FileStorage, EncryptedStorageBackend, StorageBackend};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let raw_storage = FileStorage::new(PathBuf::from("./data"));
    let key = [0u8; 32]; // Use a secure key in production!
    
    let secure_storage = EncryptedStorageBackend::new(raw_storage, key);
    
    // Transparently encrypts data before saving to disk
    secure_storage.store("secrets/key.json", b"sensitive data").await?;
    
    Ok(())
}
```

## 📂 Architecture Context
`mem-core` is the bottom-most dependency in the MindPalace DAG. All other crates (`mem-*`, `brain`) depend on `mem-core` for shared types, ensuring a strictly acyclic and modular architecture.
