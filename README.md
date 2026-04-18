# MindPalace (v0.4.0): The 8-Layer Agent Memory Ecosystem
## claude-mem in rust

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Production-Ready](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg) ![Memory Architecture: 7-Layer Relational](https://img.shields.io/badge/Architecture-7--Layer--Relational-magenta.svg)

**MindPalace** is a high-performance, production-grade memory architecture for autonomous AI agents, built entirely in Rust. It utilizes a "Defense-in-Depth" strategy with **RuVector-Graph** relational fact modeling and **HNSW** semantic search to manage massive context windows and maintain multi-session knowledge integrity.

---

## 🏗️ The 14-Crate Ecosystem

MindPalace is composed of 14 specialized crates, each handling a specific dimension of agentic memory and cognition:

| Crate | Responsibility | Core Feature |
| :--- | :--- | :--- |
| **`brain`** | **Pipeline Orchestrator** | Manages the prioritized `MemoryLayer` execution sequence. |
| **`mem-core`** | **Foundational Types** | Shared traits (`StorageBackend`, `LlmClient`) and core data models. |
| **`mem-personality`**| **Identity Guard** | Enforces persona consistency and detects "Out of Character" drift. |
| **`mem-offloader`** | **Bulk Data Management**| Prevents context overflow by offloading large tool outputs to CAS. |
| **`mem-micro`** | **Adaptive Pruning** | Relevance-based context pruning using dynamic TTL decay. |
| **`mem-session`** | **Narrative Persistence**| Periodically condenses context into technical markdown summaries. |
| **`mem-compactor`** | **Structural Defense** | Importance-based 80/20 pruning via 9-point structural models. |
| **`mem-extractor`** | **Relational Knowledge**| Distills discrete facts and resolves contradictions via LLM arbitration. |
| **`mem-retriever`** | **Cognitive Retrieval** | Sub-millisecond sub-graph retrieval using HNSW indexing. |
| **`mem-dreamer`** | **Deep Consolidation** | Background synthesis of cross-session knowledge during user idleness. |
| **`mem-bridge`** | **Multi-Agent Sync** | Efficient context freezing/forking for prompt cache optimization. |
| **`mem-broker`** | **Collective Learning** | Global fact broadcast and synchronization across agent fleets. |
| **`mem-planner`** | **Cognitive Planning** | Decomposes goals into Directed Acyclic Graphs (DAG) of tasks. |
| **`mem-resilience`**| **System Safety** | Circuit breakers and emergency JSON snapshotting on failures. |
| **`mem-viewer`** | **Telemetry UI** | Real-time glassmorphic memory dashboard (port 37777). |

---

## 🧠 The 7-Layer Hardened Pipeline

MindPalace orchestrates memory through seven distinct priority levels, ensuring the agent's context is always optimized and safe:

1.  **Identity & Offloading (P1)**: `mem-personality` anchors the agent's persona at the context base, while `mem-offloader` prunes bulky tool outputs before they hit the reasoning logic.
2.  **Adaptive Micro-Compaction (P2)**: `mem-micro` performs fine-grained pruning of recent messages based on their local relevance to the current turn.
3.  **Narrative Summarization (P3)**: `mem-session` compresses older conversation segments into high-fidelity technical narratives.
4.  **Structural Compaction (P4)**: `mem-compactor` applies a 9-point structural model (Goal, Progress, Errors, etc.) to ensure critical project state is never lost.
5.  **Fact Extraction (P5)**: `mem-extractor` identifies durable facts and commits them to the persistent Knowledge Graph.
6.  **Dream Consolidation (P6)**: `mem-dreamer` (Background) synthesizes long-term patterns and cross-session insights.
7.  **Multi-Agent Bridge (P7)**: `mem-bridge` finalizes the context for inheritance by child agents or parallel tasks.
8.  **Progressive Disclosure (PD)**: `mem-broker` tools allow agents to selectively query the graph via `search_memory` if micro-compaction is in PD mode.

---

## 📊 Observability

Integrated **Prometheus** metrics for real-time monitoring and health:

- `mindpalace_context_size_bytes`: Total serialized context footprint.
- `mindpalace_item_count`: Current number of messages in the context.
- `mindpalace_compression_ratio`: Efficiency of compaction layers.
- `mindpalace_layer_latency_seconds`: Latency profiling for every processing layer.
- `mindpalace_fact_count`: Total number of active facts in the Knowledge Graph.
- `mindpalace_total_tokens_processed`: Cumulative token throughput.

---

## 🛠️ v0.4.0 Omni-Example: Full Integration

The following example demonstrates how to initialize the full MindPalace ecosystem, including specialized layers, resilience, and background workers.

```rust
use std::sync::Arc;
use std::path::PathBuf;
use brain::Brain;
use mem_core::{MindPalaceConfig, FileStorage, MemoryRole, MemoryItem, OllamaProvider};
use mem_resilience::ResilientMemoryController;
use mem_personality::PersonalityGuard;
use mem_offloader::{ToolOffloader, OffloaderConfig};
use mem_session::SessionSummarizer;
use mem_extractor::FactExtractor;
use mem_dreamer::{DreamWorker, DreamScheduler};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Core Configuration & Backend
    let config = MindPalaceConfig::from_env();
    let storage = FileStorage::new(PathBuf::from("./storage"));
    let llm = Arc::new(OllamaProvider::new(
        "http://localhost:11434".to_string(),
        "qwen2.5-coder:7b".to_string(),
        "mxbai-embed-large".to_string(),
        Some(32768),
    ));

    // 2. Initialize the Brain (Orchestrator)
    let mut brain = Brain::new(config.clone(), None, None);

    // 3. Register Layer 1: Identity & Offloading
    brain.add_layer(Arc::new(PersonalityGuard::new(
        "You are a Senior Security Engineer specialized in Rust.".to_string(),
        Some(llm.clone()),
    )));
    brain.add_layer(Arc::new(ToolOffloader::new(
        storage.clone(),
        OffloaderConfig::default(),
    )));

    // 4. Register Layer 3: Narrative Summarization
    brain.add_layer(Arc::new(SessionSummarizer::new(
        llm.clone(),
        storage.clone(),
        config.clone(),
        "narratives".to_string(),
        true, // Validation mode enabled
    )));

    // 5. Register Layer 5: Knowledge Extraction
    let name = "security_audit_session".to_string();
    brain.add_layer(Arc::new(FactExtractor::new(
        llm.clone(),
        llm.clone(), // Embedding provider
        storage.clone(),
        config.clone(),
        "knowledge/audit_kb.json".to_string(),
        name,
    )));

    // 6. Resilience Safety Wrapper
    let brain_arc = Arc::new(brain);
    let controller = ResilientMemoryController::new(brain_arc, storage.clone(), 5);

    // 7. Initialize Background Dream Consolidation
    let dream_worker = Arc::new(DreamWorker::new(
        llm.clone(),
        storage.clone(),
        config.clone(),
        PathBuf::from("./locks/dream.lock"),
    ));
    let mut scheduler = DreamScheduler::new(dream_worker);
    scheduler.start();

    // 8. Execute Resilient Memory Processing
    let mut context = mem_core::Context::default();
    controller.optimize_resilient(&mut context).await?;

    Ok(())
}
```

---

*This ecosystem is optimized for use with the [Mentalist](https://github.com/whiterabb17/mentalist) middleware harness.*
