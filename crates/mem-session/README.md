# mem-session: Session Narrative Summarizer

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Layer: 3](https://img.shields.io/badge/Layer-3-blue.svg)

`mem-session` provides the third layer of the **MindPalace** memory pipeline. It condenses the agent's historical conversation segments into high-fidelity technical narrative summaries, ensuring that long-running sessions remain coherent without exceeding context limits.

## 🗝️ Key Features

- **Narrative Convergence**: Summarizes historical segments into concise markdown narratives that capture key decisions, progress, and context.
- **Fidelity Validation**: Unique LLM-based verification pass to ensure the generated summary accurately captures critical facts from the original history.
- **Persistent Auditability**: Automatically saves full markdown narratives to disk (default: `narrative_*.md`) for future auditing or reconstruction.
- **Strategic History Compression**: Replaces a block of original messages with a single `SESSION NARRATIVE SUMMARY` item to maximize available context tokens.
- **Customizable Compression Ratios**: Adjust how much history is summarized versus preserved in its raw form.

## 🏗️ Core Mechanism

When the context item count exceeds a configurable `summary_interval`, `mem-session`:
1. Concatenates the historical message segment.
2. Uses an LLM to generate a technical summary.
3. Validates the summary against the original content (if `validation_mode` is enabled).
4. Persists the summary to disk.
5. Injects the summary into the context and prunes the summarized messages.

## 🛠️ Usage Example

```rust
use mem_session::SessionSummarizer;
use mem_core::{Context, LlmClient, FileStorage, MindPalaceConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm: Arc<dyn LlmClient> = Arc::new(MyLlmClient::new());
    let storage = FileStorage::new(std::path::PathBuf::from("./narratives"));
    let config = MindPalaceConfig::default();
    
    let summarizer = SessionSummarizer::new(
        llm,
        storage,
        config,
        "narratives".into(),
        true // Enable validation
    );
    
    let mut context = Context::default();
    
    // Periodically condense the history into a technical session summary
    summarizer.process(&mut context).await?;
    
    Ok(())
}
```

## 📂 Architecture Context
As **Layer 3**, `mem-session` provides the primary strategy for long-term narrative persistence. It runs after micro-compaction but before fact extraction, providing a "mid-term" memory bridge for the agent.
