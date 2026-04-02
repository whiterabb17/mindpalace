# MindPalace: 7-Layer Agent Memory Ecosystem

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg) ![Memory Architecture: 7-Layer](https://img.shields.io/badge/Architecture-7--Layer-magenta.svg)

**MindPalace** is a high-performance, modular memory architecture for autonomous AI agents, built entirely in Rust. It utilizes a "Defense-in-Depth" strategy to manage large context windows, maximize prompt cache efficiency, and maintain long-term knowledge through background consolidation.

## 🧠 The 7-Layer Architecture

MindPalace organizes memory into seven distinct layers, each specialized for a specific phase of the agentic reasoning cycle:

### Layer 1: Tool Result Offloader (`mem-offloader`)
- **Role**: Prevents context overflow by offloading large tool outputs (e.g., file reads, shell results) to a Content-Addressable Storage (CAS).
- **Mechanism**: Replaces large results with a unique SHA-256 hash. The agent can "fetch" the full content only when needed.

### Layer 2: Microcompaction (`mem-micro`)
- **Role**: Immediate context pruning.
- **Mechanism**: Intelligent whitespace removal, skeletonizing of code blocks, and prioritising recent turn visibility.

### Layer 3: Session Summarizer (`mem-session`)
- **Role**: Narrative continuity.
- **Mechanism**: Performs continuous, iterative summarization of the current conversation to maintain a coherent "story" within the context window.

### Layer 4: Full Compactor (`mem-compactor`)
- **Role**: Structural context defense.
- **Mechanism**: A 9-point structural summarization that applies 80/20 pruning to history, preserving only the most critical decision points and outcomes.

### Layer 5: Fact Extractor (`mem-extractor`)
- **Role**: Durable knowledge retention.
- **Mechanism**: Distills every turn into discrete facts (intent, project settings, preferences) and persists them to `knowledge.json` with similarity-based deduplication.

### Layer 6: Background Dreamer (`mem-dreamer`)
- **Role**: Idle-time consolidation.
- **Mechanism**: A background process that triggers after 45-60 minutes of inactivity to compress old sessions using Zstd and archive them, freeing up active memory.

### Layer 7: Agent Bridge (`mem-bridge`)
- **Role**: Recursive scaling & coordination.
- **Mechanism**: Enables "Context Freezing." Sub-agents can fork from a frozen parent snapshot, inheriting an identical prefix to ensure 100% Prompt Cache hits.

## 🛠️ Usage & Integration

MindPalace is designed to be used via the **Brain** orchestrator.

```rust
use brain::Brain;
use mem_core::{Context, MemoryItem, MemoryRole};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize the Brain with defaults (all 7 layers)
    let mut brain = Brain::default();

    // 2. Prepare an agent context
    let mut context = Context {
        items: vec![MemoryItem {
            role: MemoryRole::User,
            content: "Setup the project workspace.".into(),
            timestamp: 1712076000,
            metadata: serde_json::json!({}),
        }],
    };

    // 3. Optimize context (Triggers Layers 1-4)
    brain.optimize(&mut context).await?;

    // 4. Persistence is handled automatically in the background
    Ok(())
}
```

## 📂 Repository Structure

- `crates/mem-core`: Shared types and filesystem storage traits.
- `crates/brain`: The primary API and orchestrator for all layers.
- `crates/mem-*`: Specialized implementations for each of the 7 layers.

---

*Pairs perfectly with the [Mentalist](file:///d:/Repos/MindPalace/mentalist/README.md) middleware harness.*
