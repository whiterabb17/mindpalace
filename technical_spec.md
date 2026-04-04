# MindPalace Memory: Technical Implementation Specification

This document defines the production-hardened implementation details of the 7-layer memory architecture used in the `mentalist` agent system.

---

## 1. Core Memory Abstractions

### `Arc<Context>` Optimization
All context management now operates on an **Atomic Reference Counted** shared pointer.
- **`mentalist::Request`**: Contains `pub context: Arc<Context>`.
- **Memory items**: `pub items: Vec<MemoryItem>`.
- **CoW Memory**: Mid-session mutations (Compaction/Enrichment) must clone the inner context before modification:
  ```rust
  let mut current_ctx = (*self.state.context).clone();
  current_ctx.items.push(new_item);
  self.state.context = Arc::new(current_ctx);
  ```

---

## 2. Traits and Orchestration

### `MemoryLayer` (Crate: `mem-core`)
A synchronous/asynchronous trait for specialized optimization layers.
```rust
#[async_trait]
pub trait MemoryLayer: Send + Sync {
    fn name(&self) -> &str;
    async fn process(&self, context: &mut Context) -> anyhow::Result<()>;
    fn priority(&self) -> u32;
}
```

### `Brain` (Crate: `brain`)
Orchestrates the prioritized sequence of `MemoryLayer`s. Features include:
- **Prioritized Pipeline**: Sorts layers by `priority()`.
- **Metrics Integration**: Prometheus-based latency and compression tracking.
- **Budget Enforcement**: Hard-limit protection (Pruning) as a fallback.

### `ResilientMemoryController` (Crate: `mem-resilience`)
A fault-tolerant wrapper for the `Brain`.
- **Failure Thresholds**: Tracks consecutive layer failures and triggers early returns.
- **Atomic IO**: Enforces a temporary-and-rename strategy for all session persistence.

---

## 3. Sandboxing & Safety (`mentalist::executor`)

The memory system is decoupled from execution but depends on its safety guarantees.
- **`CommandValidator`**: Features a strict whitelist of allowed commands (e.g., `ls`, `cat`, `python`).
- **Path Canonicalization**: Prevents directory traversal in the vault or sandbox root.
- **Wasm/Docker Modes**: Provides isolated environments for high-risk tool results (Layer 1 Offloading).

---

## 4. Layer Detail (Production Crates)

| Layer | Implementation Crate | Primary Trigger |
| :--- | :--- | :--- |
| **L1: Offloader** | `mem-offloader` | Item content > 2,048 bytes |
| **L2: Microcompaction** | `mem-compactor` | Context > 20% model limit |
| **L3: Session Log** | `mem-session` | Intervals (5 turns) |
| **L4: Full Compaction** | `mem-compactor` | Context > 80% model limit |
| **L5: Fact Extraction** | `mem-extractor` | `after_ai_call` hook (Final Chunk) |
| **L6: Dreaming** | `mem-dreamer` | Background PID-locked idle scan |
| **L7: Broker** | `mem-broker` | Cross-agent delegation (IPC) |

---

## 5. Middleware Integration (`mentalist::harness`)

The `Harness` acts as the primary integration point, allowing `Middleware` to trigger memory optimization:
- **`before_ai_call`**: Context retrieval (RAG) and Layer 2 optimization.
- **`after_ai_call`**: Result accumulation and Layer 5 extraction.
- **Streaming Support**: Guaranteed hook execution via the `run_stream` wrapper.

---

## 6. Persistence Strategy

All persistent memory artifacts follow a strict durability model:
1.  **Stage**: Write to `.agent/tmp/<id>.json`.
2.  **Verify**: Ensure JSON is valid and checksum matches.
3.  **Commit**: Rename (atomic on POSIX/Windows) to `.agent/knowledge.json`.
4.  **Prune**: Cleanup orphaned temporary files.
