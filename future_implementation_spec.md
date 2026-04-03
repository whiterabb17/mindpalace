# Future Implementation Specification: RuVector Integration

This document outlines the planned migration from the initial simple vector storage to the **RuVector** ecosystem for high-performance memory management.

## Why RuVector?
The current implementation uses a simple `Vec<(Embedding, FactId)>` which is sufficient for small-scale knowledge bases. However, as the `mindpalace` grows, we will require:
- **Scalability**: Sub-millisecond search across millions of facts.
- **HNSW Indexing**: Efficient similarity search that bypasses O(n) linear scans.
- **Relational Memory**: Leveraging `ruvector-graph` to navigate fact dependencies and provenance more naturally than a flat list.
- **Quantization**: Reduced memory footprint for high-dimensional embeddings.

## Target Architecture

### Core Components
1. **`ruvector-core`**: Will replace the internal `VectorIndex`.
   - Implement HNSW indexing for durable facts.
   - Use SIMD acceleration for distance calculations.
2. **`ruvector-graph`**: Will replace the `FactGraph` flat storage.
   - Store `FactNode` as graph vertices.
   - Use edges to represent `superseded_by`, `dependencies`, and `source_session_id`.
   - Enables Cypher-style queries like: "Find all facts related to [Project X] that were superseded in the last 24 hours."

### Integration Strategy

#### Phase 1: Storage Layer Abstraction
Decouple the `MemoryRetriever` from the underlying storage by defining a generic `VectorStore` trait.
```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn insert(&self, id: &str, embedding: &[f32], metadata: FactNode) -> anyhow::Result<()>;
    async fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(FactNode, f32)>>;
}
```

#### Phase 2: RuVector Backend Implementation
Implement the `VectorStore` trait using `ruvector-core`.
- Initialize a local `ruvector` collection in `~/.mindpalace/vectors/`.
- Map `FactNode` metadata into the collection schema.

#### Phase 3: Graph-Augmented Retrieval
Enhance the `MemoryRetriever` to use `ruvector-graph`.
- Instead of simple similarity, use graph traversals to find non-superseded, high-confidence "clusters" of related knowledge.

## Dependencies to Add
- `ruvector-core`
- `ruvector-graph`
- `ruvector-collections` (for multi-tenant or multi-agent support)

## Migration Path
1. Export all current `FactNode` items from SQL/JSON.
2. Generate/Validate embeddings for all items.
3. Batch-insert into `ruvector` collection.
4. Verify retrieval fidelity against the legacy `Vec`-based implementation.
