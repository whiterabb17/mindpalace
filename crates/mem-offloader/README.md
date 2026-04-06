# mem-offloader: Tool Result Offloader

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Layer: 1](https://img.shields.io/badge/Layer-1-blue.svg)

`mem-offloader` is the first layer in the **MindPalace** memory pipeline. Its primary mission is to prevent context overflow by identifying and offloading exceptionally large message contents (typically tool outputs or long assistant responses) to external storage.

## 🗝️ Key Features

- **Content-Addressable Storage (CAS)**: Uses **SHA-256** hashing to generate unique identifiers for offloaded content, ensuring data integrity and enabling automatic deduplication.
- **Dynamic Thresholding**: Configurable character length limits determine when an item should be offloaded (default: 2048 chars).
- **Context Preservation**: Replaces the bulky original content with a lightweight "stub" containing the hash and a short, Unicode-safe preview (default: 100 chars).
- **Audit-Ready Metadata**: Injects `offloaded` and `storage_id` markers into the `MemoryItem` metadata for future retrieval or auditing.
- **Role-Specific Targeting**: Specifically monitors `Assistant` and `Tool` roles, which are the most common sources of large data chunks.

## 🏗️ Core Mechanism

When a message exceeds the configured `threshold`, `mem-offloader`:
1. Calculates the SHA-256 hash of the content.
2. Persists the full content to the configured `StorageBackend` (e.g., `FileStorage`).
3. Updates the `MemoryItem` with a preview stub and metadata links.

## 🛠️ Usage Example

```rust
use mem_offloader::{ToolOffloader, OffloaderConfig};
use mem_core::{FileStorage, Context, MemoryItem, MemoryRole};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let storage = FileStorage::new(PathBuf::from("./blobs"));
    let config = OffloaderConfig {
        threshold: 1024,
        preview_len: 50,
    };
    
    let offloader = ToolOffloader::new(storage, config);
    let mut context = Context::default();
    
    // Process context to offload massive tool results
    offloader.process(&mut context).await?;
    
    Ok(())
}
```

## 📂 Architecture Context
As **Layer 1**, `mem-offloader` runs before any summarization or extraction layers. This ensures that downstream layers (which may involve LLM calls) are not overwhelmed by massive, non-semantic raw data.
