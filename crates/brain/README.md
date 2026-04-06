# Brain: Core Pipeline Orchestrator

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Production](https://img.shields.io/badge/Status-Stable-green.svg)

The `Brain` crate is the central execution engine of the **MindPalace** memory architecture. It coordinates the registration, prioritization, and execution of various memory layers (`MemoryLayer`) in a single, cohesive processing pipeline.

## 🗝️ Key Features

- **Layered Processing**: Executes memory layers (e.g., pruning, summarization, extraction) in a deterministic, priority-ordered sequence.
- **Budget Enforcement**: Automatically prunes context if it exceeds the configured maximum item limit (protective "emergency pruning").
- **Performance Profiling**: Integrated Prometheus metrics for measuring layer-by-layer latency, context size, and compression ratios.
- **Token Efficiency Tracking**: Calculates token usage before and after optimization to monitor the real-world compression performance.
- **Dynamic Configuration**: Easily adjustable limits and thresholds via the `MindPalaceConfig`.

## 🏗️ Core Mechanism

The `Brain` operates on a provided `Context`, applying each registered `MemoryLayer` in order of its priority (lowest priority value executes first).

```rust
// Priority Execution Sequence
// 1. mem-personality (Priority 1) -> Persona Alignment
// 2. mem-offloader (Priority 2)   -> Large Output Offloading
// 3. mem-micro (Priority 3)       -> Quick Pruning & TTL
// ...
// 7. mem-bridge (Priority 7)      -> Multi-agent Freezing
```

## 🛠️ Usage Example

```rust
use brain::Brain;
use mem_core::{Context, MindPalaceConfig, MemoryLayer};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = MindPalaceConfig::default();
    let mut brain = Brain::new(config, None, None);

    // Register active memory layers
    // brain.add_layer(Arc::new(PersonalityGuard::new(...)));
    // brain.add_layer(Arc::new(AdaptiveMicroCompactor::new(...)));

    let mut context = Context::default();

    // Execute the full 7-layer memory optimization cycle
    brain.optimize(&mut context).await?;

    Ok(())
}
```

## 📂 Architecture Context
The `Brain` acts as the primary interface for any **MindPalace** installation. It abstracts away the complexity of layer-by-layer coordination, providing a simple `optimize` call for the agent to use during its turn.
