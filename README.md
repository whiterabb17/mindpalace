# MindPalace (v0.2.0): Hardened 7-Layer Agent Memory Ecosystem

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Production-Ready](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg) ![Memory Architecture: 7-Layer Relational](https://img.shields.io/badge/Architecture-7--Layer--Relational-magenta.svg)

**MindPalace** is a high-performance, production-grade memory architecture for autonomous AI agents, built entirely in Rust. It utilizes a "Defense-in-Depth" strategy with **RuVector-Graph** relational fact modeling and **HNSW** semantic search to manage massive context windows and maintain multi-session knowledge integrity.

## 🧠 The 7-Layer Hardened Pipeline

MindPalace orchestrates memory into seven distinct layers, specialized for the agentic reasoning cycle:

### Layer 1: Tool Result Offloader (`mem-offloader`)
- **Role**: Prevents context overflow by offloading large tool outputs to Content-Addressable Storage (CAS).
- **Relational Link**: Each offloaded item is tracked with session provenance and a SHA-256 identifier.

### Layer 2: Adaptive Micro-Compactor (`mem-micro`)
- **Role**: Dynamic relevance-based context pruning.
- **Mechanism**: Prunes context items using a `RelevanceAnalyzer`. Adjusts message longevity through Linear, Exponential, or Adaptive TTL decay.

### Layer 3: Session Narrative Summarizer (`mem-session`)
- **Role**: Context Compression & Narrative Persistence.
- **Mechanism**: PERIODICALLY condenses conversation segments into technical markdown summaries. Persists narratives to persistent disk for audit and cold-start reconstruction.

### Layer 4: Intelligent Full Compactor (`mem-compactor`)
- **Role**: Structural context defense.
- **Mechanism**: Uses importance-based 80/20 pruning via 9-point structural summarization (Goal, Progress, Errors, Next). Creates safety checkpoints before every compaction.

### Layer 5: Fact Extraction & Knowledge Graph (`mem-extractor`, `mem-core`)
- **Role**: Relational Knowledge Retention.
- **Mechanism**: Distills discrete facts and persists them to `ruvector-graph`. Automatically resolves contradictions via the **ConflictResolver** with LLM arbitration.

### Layer 6: HNSW Memory Retriever (`mem-retriever`)
- **Role**: Graph-Augmented RAG.
- **Mechanism**: Uses `ruvector-core` with HNSW indexing for sub-millisecond sub-graph retrieval. Synchronizes volatile semantic indexes with persistent relational storage during hydration.

### Layer 7: Dream Worker Consolidation (`mem-dreamer`)
- **Role**: Deep-Review & Maintenance.
- **Mechanism**: Background task that synthesizes cross-session knowledge during user idleness. Persists high-level structural patterns to durable synthesis files.

## 🛡️ Resilience & Safety (`mem-resilience`)

MindPalace includes a dedicated resilience layer to protect agents from infrastructure failure:
- **Circuit Breaker**: Prevents "death spirals" by cutting LLM/storage operations if failure thresholds are reached.
- **Emergency Snapshotting**: Automatically creates forensics-ready JSON snapshots of the agent's context on any optimization failure.

## 📊 Observability

Integrated **Prometheus** metrics for real-time monitoring:
- `mindpalace_context_size_bytes`: Current memory footprint.
- `mindpalace_compression_ratio`: Efficiency of the compaction layers.
- `mindpalace_layer_latency_seconds`: Performance profiling for each processing layer.
- `mindpalace_total_tokens_processed`: Cumulative token throughput.

## 🛠️ Usage Example

```rust
use brain::Brain;
use mem_resilience::ResilientMemoryController;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let brain = Arc::new(Brain::new(Some(metrics), Some(token_counter)));
    let controller = ResilientMemoryController::new(brain, storage, 5);

    // Optimized memory processing with circuit-breaker protection
    controller.optimize_resilient(&mut context).await?;

    Ok(())
}
```

## 📂 Repository Structure

- `crates/mem-core`: Shared types, **GraphDB** interfaces, and serialization.
- `crates/mem-extractor`: Fact distillation and conflict resolution logic.
- `crates/mem-retriever`: **HNSW** semantic search and graph-augmented RAG.
- `crates/mem-resilience`: Circuit-breakers and snapshotting for memory stability.
- `crates/mem-micro`: Adaptive TTL and relevance-based pruning.
- `crates/mem-session`: Narrative summarization and history persistence.
- `crates/mem-compactor`: Structural importance-based summarization.
- `crates/mem-dreamer`: Background idle-triggered consolidation.
- `crates/brain`: Core pipeline orchestrator.

---

*This ecosystem is optimized for use with the [Mentalist](https://github.com/whiterabb17/mentalist/README.md) middleware harness.*
