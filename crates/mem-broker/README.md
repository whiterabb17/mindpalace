# mem-broker: Multi-Agent Fact Broker

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Stable](https://img.shields.io/badge/Status-Stable-green.svg)

`mem-broker` provides the collective learning and knowledge-sharing infrastructure for the **MindPalace** memory ecosystem. It acts as a central broker for broadcasting and synchronizing facts between multiple agents in a fleet or project.

## 🗝️ Key Features

- **Collective Learning**: Enables agents to share high-confidence technical or objective facts with a global pool for project-wide synchronization.
- **High-Confidence Filtering**: Automatically filters facts by confidence score (>= 0.9) and scope (`Global` or `Project`) before broadcasting.
- **Atomic Synchronization**: Uses a file-based global lock to ensure safe, atomic updates to the shared knowledge pool from multiple agents.
- **Pull-Based Integration**: Allows agents to pull shared knowledge from a common backend (e.g., S3, Shared Drive) and integrate it into their local fact graphs.
- **Agent Interoperability**: Decouples knowledge producers from consumers, providing a standardized, structured JSON-based fact format for fleet communication.

## 🏗️ Core Mechanism

The `FactBroker` handles the exchange of facts between agents:
1. **Publish**: Scans a provided list of `FactNode` items, filters by confidence and scope, acquires a lock, and persists them to shared storage.
2. **Pull**: Scans the shared storage for all available facts and returns them for local reconciliation and integration.

## 🛠️ Usage Example

```rust
use mem_broker::FactBroker;
use mem_core::{FileStorage, FactNode, FactScope};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let shared_storage = FileStorage::new(PathBuf::from("./shared_kb"));
    let broker = FactBroker::new(shared_storage, PathBuf::from("/tmp/broker.lock"));
    
    // Publish high-confidence facts from the local session
    let fact = FactNode::new("Target project uses Rust v1.75".into(), "Technical".into(), 0.95, "session-123".into());
    broker.publish_facts(vec![fact]).await?;
    
    // Pull the latest shared knowledge from other agents in the fleet
    let fleet_knowledge = broker.pull_shared_knowledge().await?;
    
    Ok(())
}
```

## 📂 Architecture Context
`mem-broker` is a standalone service in the MindPalace ecosystem. It is typically used during agent handovers or at the end of a session to ensure the entire agent fleet benefits from the knowledge extracted during the reasoning cycle.
