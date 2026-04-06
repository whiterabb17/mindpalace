# mem-micro: Adaptive Micro-Compactor

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Layer: 2](https://img.shields.io/badge/Layer-2-blue.svg)

`mem-micro` provides the second layer of the **MindPalace** memory pipeline. It focuses on high-frequency, low-latency context pruning by evaluating the "stickiness" of individual memory items using age, relevance, and role.

## 🗝️ Key Features

- **Relevance-Based Pruning**: Unlike simple FIFO buffers, `mem-micro` uses a `RelevanceAnalyzer` to provide "stickiness" to important recent messages.
- **Adaptive TTL Decay**: Dynamically calculates the Time-To-Live (TTL) for each item using three available strategies:
  - **Linear**: Fixed-slope decay based on message age.
  - **Exponential**: Half-life-based decay for long-running sessions.
  - **Adaptive**: Boosts effective TTL by up to 3x based on local relevance scores.
- **Role-Aware Security**: Automatically prevents the accidental pruning of **System** messages ("Immortal Context").
- **High Performance**: Optimized for rapid execution between model turns to maintain a lean active context.

## 🏗️ Core Mechanism

The `AdaptiveMicroCompactor` iterates through the context, calculating an `effective_ttl` for each non-system item:

```rust
// TTL = base_ttl * (1.0 + relevance_score)
// If age > TTL, the item is pruned from the context.
```

## 🛠️ Usage Example

```rust
use mem_micro::{AdaptiveMicroCompactor, TTLDecayStrategy};
use mem_core::{Context, MindPalaceConfig, RelevanceAnalyzer};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = MindPalaceConfig::default();
    let analyzer = Arc::new(MockRelevanceAnalyzer::new()); // Replace with a real analyzer
    
    let compactor = AdaptiveMicroCompactor::new(
        config,
        TTLDecayStrategy::AdaptiveByType,
        analyzer
    );
    
    let mut context = Context::default();
    
    // Prune stale, low-relevance items before the next agent turn
    compactor.process(&mut context).await?;
    
    Ok(())
}
```

## 📂 Architecture Context
As **Layer 2**, `mem-micro` handles the "quiet noise" of a conversation. It runs after offloading but before full summarization, keeping the context size within manageable bounds for more intensive downstream processing.
