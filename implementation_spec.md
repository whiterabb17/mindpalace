# Agent Memory Crate: Implementation Specification

This specification outlines the technical design for a Rust crate that implements the 7-layer memory architecture for autonomous AI agents.

## Core Concepts

### `MemoryItem`
The atomic unit of memory, which can be a User Message, LLM Response, Tool Call, or Tool Result.

### `Context`
The active, in-memory window of `MemoryItem`s currently being sent to the LLM.

---

## 1. Traits and Abstractions

### `MemoryLayer`
Each layer implements this trait to handle context optimization.
```rust
pub trait MemoryLayer {
    fn name(&self) -> &str;
    fn process(&mut self, context: &mut Context) -> anyhow::Result<()>;
    fn priority(&self) -> u32;
}
```

### `StorageBackend`
Abstracting where physical data (Tool Results, Knowledge Files) is stored.
```rust
pub trait StorageBackend {
    fn store(&self, id: &str, data: &[u8]) -> anyhow::Result<()>;
    fn retrieve(&self, id: &str) -> anyhow::Result<Vec<u8>>;
    fn exists(&self, id: &str) -> bool;
}
```

### `Summarizer`
Interface for LLM-driven condensation.
```rust
pub trait Summarizer {
    async fn summarize_session(&self, items: Vec<MemoryItem>) -> anyhow::Result<String>;
    async fn full_compaction(&self, items: Vec<MemoryItem>) -> anyhow::Result<String>;
}
```

---

## 2. Layer Implementation Details

### Layer 1: Tool Result Storage (`ToolOffloader`)
- **Threshold**: Configurable (default 2,048 bytes).
- **Behavior**: If `item.size() > threshold`, move content to `StorageBackend`. Replace `item.content` with `[Output Stored: <id>. Preview: <first 100 bytes>...]`.

### Layer 2: Microcompaction (`MicroCompactor`)
- **TTL**: Items expire after $X$ seconds or $Y$ interactions.
- **Cache Logic**: Designed to use `cache_edits` APIs (if supported by the provider) to remove middle items without shifting the total prefix tokens for subsequent items.

### Layer 3: Session Memory (`SessionRecorder`)
- **Interval**: Every $N$ tokens or $M$ tool calls.
- **Output**: A `markdown` file that acts as a "running log" of accomplishments and current state.

### Layer 4: Full Compaction (`HardLimitCompactor`)
- **Limit**: $80\%$ of model's context window.
- **Strategy**: 9-Point Summarization (Goal, Context, Tools Used, Critical Failures, Current Progress, Pending Tasks, Constants, User Preferences, Next Step).

### Layer 5: Knowledge Extraction (`FactExtractor`)
- **Persistence**: Writes to `.agent/memory/knowledge.json`.
- **Fields**: `user_intent`, `project_context`, `habitual_workflows`.

### Layer 6: Dreaming (`BackgroundDreamer`)
- **Concurrency**: Requires a background task.
- **Locking**: Uses a PID-file (e.g., `.agent/memory/dreaming.lock`) to prevent multiple agents from "dreaming" on the same memory store simultaneously.
- **Executor**: Default implementation uses `tokio::spawn`.

### Layer 7: Cross-Agent Comms (`AgentBridge`)
- **Protocol**: JSON-RPC or a simple IPC pipe.
- **Shared Storage**: Uses a content-addressable storage (CAS) for messages to avoid duplication between parent/child instances.

---

## 3. Recommended Dependency Stack

- **`tokio`**: For async runtime and background tasks.
- **`anyhow` / `thiserror`**: For robust error handling.
- **`serde` / `serde_json`**: For persistence serialization.
- **`tracing`**: For observability of memory compaction events.
- **`fs4`**: For cross-platform file locking (PID-level mutex).
- **`sha2`**: For content-addressable storage of tool results.

## 4. Example Usage (Pseudocode)

```rust
let mut memory = MemoryManager::builder()
    .with_layer(ToolOffloader::new(2048))
    .with_layer(MicroCompactor::default())
    .with_summarizer(MyLlmSummarizer::new(api_key))
    .build();

// During agent loop:
let response = agent.run_next().await?;
memory.add_item(response).await?;
memory.optimize().await?; // Triggers layers based on priority and context state
```
