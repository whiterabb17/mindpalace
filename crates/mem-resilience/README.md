# mem-resilience: Resilience & Safety Layer

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Production](https://img.shields.io/badge/Status-Hardened-green.svg)

`mem-resilience` provides critical safety and reliability mechanisms for the **MindPalace** memory ecosystem. It protects agents from infrastructure failures, LLM outages, and cascading errors during high-frequency memory cycles.

## 🗝️ Key Features

- **Circuit Breaker Protection**: Monitors for repeated failures in backend services (LLMs, Storage) and enters an `Open` state to save error budget and prevent "death spirals."
- **Emergency Snapshotting**: Automatically creates forensics-ready JSON snapshots of the agent's context (`emergency/snapshot_*.json`) whenever an optimization cycle fails.
- **Graceful Degradation**: Detects when the circuit is open and automatically bypasses non-essential memory operations, ensuring the agent remains functional during partial outages.
- **State Management**: Implements standard `Closed`, `Open`, and `HalfOpen` states with configurable failure thresholds and recovery timeouts.
- **Resilient Orchestration**: Provides the `ResilientMemoryController`, a production-grade wrapper for the `Brain` that integrates both performance and resilience policies.

## 🏗️ Core Mechanism

The `ResilientMemoryController` acts as a protective proxy for the `Brain`:
1. **Health Check**: Queries the `CircuitBreaker` state.
2. **Execution**: Attempts `brain.optimize()`.
3. **Success**: Reports success to the breaker and resets failure counters.
4. **Failure**: Reports failure, trips the breaker if thresholds are met, and creates an emergency context recovery file.

## 🛠️ Usage Example

```rust
use mem_resilience::{ResilientMemoryController, CircuitBreaker};
use brain::Brain;
use mem_core::{Context, FileStorage, MindPalaceConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let brain = Arc::new(Brain::default());
    let storage = FileStorage::new(std::path::PathBuf::from("./data"));
    
    // Failure threshold: 5, Reset timeout: 30s
    let controller = ResilientMemoryController::new(brain, storage, 5);
    
    let mut context = Context::default();
    
    // Execute memory optimization with automatic circuit-breaker and snapshotting
    controller.optimize_resilient(&mut context).await?;
    
    Ok(())
}
```

## 📂 Architecture Context
`mem-resilience` is the "safety harness" for MindPalace. It is typically used at the highest level of an agent application, wrapping the core memory engine to ensure production stability and auditability.
