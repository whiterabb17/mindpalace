# mem-dreamer: Dream Worker Consolidation

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Layer: 7](https://img.shields.io/badge/Layer-7-blue.svg)

`mem-dreamer` provides the seventh and final layer of the **MindPalace** memory pipeline. It focuses on the "offline" deep-review and synthesis of historical session data, performing memory consolidation during user idleness to maintain long-term knowledge integrity.

## 🗝️ Key Features

- **Background Memory Synthesis**: Performs deep analysis of historical session JSON files to identify goal patterns, technical constraints, and evolving facts.
- **Structural Knowledge Consolidation**: Transforms disparate session history into durable, high-level knowledge markdown files (`knowledge/synthesis_*.md`).
- **Idle-Triggered Scheduling**: Integrated `DreamScheduler` that monitors user interaction and automatically triggers consolidation cycles after a configurable idle threshold (e.g., 60 minutes).
- **Fleet-Wide Resilience**: Uses file-based exclusive locking to prevent concurrent consolidation attempts across multiple agent instances or processes.
- **Durable Synthesis Persistence**: Provides high-quality, pre-processed input for **Layer 6** (retrieval), ensuring the agent has access to consolidated patterns and high-level summaries.
- **Offline Efficiency**: Operates entirely in the background, ensuring zero latency impact on the agent's real-time decision-making loop.

## 🏗️ Core Mechanism

The `DreamWorker` operates outside the real-time context pipeline:
1. **Polling**: `DreamScheduler` monitors elapsed time since the last `record_activity` call.
2. **Locking**: Worker attempts to acquire a global lock before processing history.
3. **Consolidation**: LLM reviews historical sessions to find long-term patterns and structural knowledge.
4. **Persistence**: Consolidated facts and narratives are stored as durable markdown files.

## 🛠️ Usage Example

```rust
use mem_dreamer::{DreamWorker, DreamScheduler};
use mem_core::{LlmClient, FileStorage, MindPalaceConfig};
use std::sync::Arc;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm: Arc<dyn LlmClient> = Arc::new(MyLlmClient::new());
    let storage = FileStorage::new(PathBuf::from("./data"));
    let config = MindPalaceConfig::default();
    
    let worker = Arc::new(DreamWorker::new(
        llm,
        storage,
        config,
        PathBuf::from("/tmp/dream.lock")
    ));
    
    let mut scheduler = DreamScheduler::new(worker);
    
    // Start the background consolidation scheduler
    scheduler.start();
    
    // In your main loop, update the scheduler on user activity
    // scheduler.record_activity();
    
    Ok(())
}
```

## 📂 Architecture Context
As **Layer 7**, `mem-dreamer` is the final step in the agent's reasoning cycle. It provides periodic knowledge consolidation that enhances the long-term relevance and reliability of the internal fact graph managed by earlier layers.
