# mem-compactor: Intelligent Full Compactor

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Layer: 4](https://img.shields.io/badge/Layer-4-blue.svg)

`mem-compactor` provides the fourth layer of the **MindPalace** memory pipeline. It is a heavy-duty compression engine that activates when the agent's context exceeds its maximum capacity, performing structural summarization and importance-based pruning.

## 🗝️ Key Features

- **80/20 Importance-Based Pruning**: Uses an `ImportanceAnalyzer` to score every context item. It retains the top 1/3 most critical messages in their raw form while summarizing the rest.
- **9-Point Structural Summarization**: Condenses historical context into a precise, 9-point technical state:
  1. **GOAL**: Original user objective.
  2. **CONTEXT**: Environmental facts.
  3. **TOOLS**: Successfully used tools.
  4. **ERRORS**: Technical failures informing the current approach.
  5. **PROGRESS**: Percentage towards goal.
  6. **PENDING**: Immediate next steps.
  7. **CONSTANTS**: Immutable facts/constraints.
  8. **PREFERENCES**: User formatting/style preferences.
  9. **NEXT**: The very next action.
- **Safety Checkpointing**: Automatically creates a full serialized JSON backup of the context (`checkpoint_*.json`) before performing any destructive transformations.
- **Ordered Retention**: Maintains the chronological sequence of high-importance items alongside the new structural summary to preserve conversational flow.

## 🏗️ Core Mechanism

When the context limit is reached, `mem-compactor`:
1. Creates a safety checkpoint in the configured storage.
2. Scores all items using the `ImportanceAnalyzer`.
3. Splits items into "Keep" (High Importance) and "Summarize" (Low Importance) sets.
4. Generates a structural summary of the "Summarize" set.
5. Reconstructs the context with the summary and the ordered "Keep" set.

## 🛠️ Usage Example

```rust
use mem_compactor::IntelligentFullCompactor;
use mem_core::{Context, LlmClient, ImportanceAnalyzer, FileStorage, MindPalaceConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm: Arc<dyn LlmClient> = Arc::new(MyLlmClient::new());
    let analyzer: Arc<dyn ImportanceAnalyzer> = Arc::new(MyImportanceAnalyzer::new());
    let storage = FileStorage::new(std::path::PathBuf::from("./checkpoints"));
    let config = MindPalaceConfig::default();
    
    let compactor = IntelligentFullCompactor::new(
        llm,
        analyzer,
        storage,
        config,
        "checkpoints".into()
    );
    
    let mut context = Context::default();
    
    // Perform structural compaction when context is nearly full
    compactor.process(&mut context).await?;
    
    Ok(())
}
```

## 📂 Architecture Context
As **Layer 4**, `mem-compactor` is the "last line of defense" for context management. It runs after filtering and summarization layers, ensuring that the agent always has a structured, high-fidelity view of its long-term objectives even in massive sessions.
