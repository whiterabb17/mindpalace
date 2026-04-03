# mem-core (v0.2.0)

The foundation of the MindPalace memory ecosystem. This crate defines the core data models, traits, and storage interfaces used by all other memory layers.

## Key Components

### `FactNode`
A structured representation of a single unit of knowledge.
- **Relational Metadata**: Supports `superseded_by` and `dependencies` fields for graph modeling.
- **Semantic Metadata**: Optional `embedding` field for vector search integration.
- **Provenance**: Tracks source session IDs and timestamps.

### `FactGraph`
An operational handle for **RuVector-Graph** (`GraphDB`).
- **Persistence**: Manages vertex and edge creation for relational fact storage.
- **Querying**: Provides abstractions for fetching non-superseded facts and navigating dependency chains.

### `KnowledgeBase`
The top-level container that encapsulates the `FactGraph`.

### `StorageBackend`
A trait for pluggable persistence (e.g., `FileStorage`).

### Middleware Traits
- `MemoryLayer`: Standard interface for all pipeline stages.
- `ModelProvider`: Interface for LLM completions and streaming.
- `TokenCounter`: Standard for token management.
- `EmbeddingProvider`: Standard for vector generation.

## Relational Schema (GraphDB)

Nodes are labeled as `:Fact` and contain properties matching the `FactNode` struct. Relationships include:
- `[:SUPERSEDED_BY]`: Points from an old fact to its replacement.
- `[:DEPENDS_ON]`: Indicates a logical dependency between facts.

## Usage

```rust
use mem_core::{FactNode, FactGraph};

let graph = FactGraph::new(Some("./storage/graph.db".into()))?;
let fact = FactNode::new("User prefers dark mode".into(), "Preferences".into(), 1.0, "session_001".into());
graph.add_fact(fact)?;
```
