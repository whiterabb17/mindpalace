# mem-bridge: Agent Context Bridge

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Stable](https://img.shields.io/badge/Status-Stable-green.svg)

`mem-bridge` provides the multi-agent orchestration and context persistence capabilities for the **MindPalace** memory ecosystem. It enables agents to "freeze" their current state into snapshots and "fork" new contexts for child agents or recovery tasks.

## 🗝️ Key Features

- **Context Freezing**: Persists the entire agent context into a byte-identical JSON snapshot, ensuring precise state preservation for audits or recovery.
- **Agent Inheritance**: Facilitates context forking, allowing child agents to inherit a parent's prefix (improving prompt cache efficiency and performance).
- **Metadata-Driven Snapshots**: Automatically triggers freezes based on specific context metadata (`freeze_trigger`), enabling dynamic, rule-based state preservation.
- **Disaster Recovery**: Simplifies session restoration by allowing agents to re-hydrate their context from any previously frozen snapshot.
- **Async Safety**: High-performance, asynchronous implementation for seamless integration with real-time agentic loops.

## 🏗️ Core Mechanism

The `AgentBridge` manages the lifecycle of context snapshots:
1. **Freeze**: Serializes and stores the current `Context` with a unique `snapshot_id`.
2. **Fork**: Retrieves and deserializes a snapshot to bootstrap a new `Context` instance.
3. **Trigger**: (MemoryLayer) Monitors the message stream for metadata flags to automate the snapshot process.

## 🛠️ Usage Example

```rust
use mem_bridge::AgentBridge;
use mem_core::{FileStorage, Context, MemoryLayer};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let storage = FileStorage::new(std::path::PathBuf::from("./snapshots"));
    let bridge = AgentBridge::new(storage);
    
    let context = Context::default();
    
    // Freeze current context for child agent inheritance
    bridge.freeze_context("task-alpha-base", &context).await?;
    
    // Fork a new context from the snapshot for a parallel reasoning sub-task
    let forked_context = bridge.fork_context("task-alpha-base").await?;
    
    Ok(())
}
```

## 📂 Architecture Context
`mem-bridge` is the "connector" in the MindPalace DAG. It is typically implemented as the final layer in the pipeline (Priority 7), ensuring that the agent's optimized and compacted context is what gets persisted for inheritance or recovery.
