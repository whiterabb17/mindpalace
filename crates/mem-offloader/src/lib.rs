use mem_core::{MemoryLayer, Context, StorageBackend, MemoryRole};
use async_trait::async_trait;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloaderConfig {
    /// Token length threshold for offloading (default: 2048 characters)
    pub threshold: usize,
    /// Length of the preview to keep in context (default: 100 characters)
    pub preview_len: usize,
}

impl Default for OffloaderConfig {
    fn default() -> Self {
        Self {
            threshold: 2048,
            preview_len: 100,
        }
    }
}

pub struct ToolOffloader<S: StorageBackend> {
    pub storage: S,
    pub config: OffloaderConfig,
}

impl<S: StorageBackend> ToolOffloader<S> {
    pub fn new(storage: S, config: OffloaderConfig) -> Self {
        Self { storage, config }
    }
}

#[async_trait]
impl<S: StorageBackend> MemoryLayer for ToolOffloader<S> {
    fn name(&self) -> &str {
        "ToolOffloader"
    }

    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        for item in &mut context.items {
            // Memory conservation: Only offload tool or assistant outputs which tend to be the bulkiest
            match item.role {
                MemoryRole::Assistant | MemoryRole::Tool => {
                    if item.content.len() > self.config.threshold {
                        // 1. Content Addressable Hashing
                        let mut hasher = Sha256::new();
                        hasher.update(item.content.as_bytes());
                        let hash = format!("{:x}", hasher.finalize());
                        let id = format!("blobs/{}", hash);

                        // 2. Persist to storage (Shared Backend)
                        self.storage.store(&id, item.content.as_bytes()).await?;

                        // 3. Create context-friendly stub (Unicode-safe preview)
                        let preview: String = item.content.chars().take(self.config.preview_len).collect();
                        item.content = format!(
                            "[Large Output Offloaded to Storage. Hash: {}. Preview: {}...]",
                            hash, preview
                        );
                        
                        // 4. Update metadata for auditability
                        let mut meta = item.metadata.as_object_mut()
                            .cloned()
                            .unwrap_or_default();
                        meta.insert("offloaded".to_string(), serde_json::Value::Bool(true));
                        meta.insert("storage_id".to_string(), serde_json::Value::String(id));
                        item.metadata = serde_json::Value::Object(meta);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn priority(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mem_core::{FileStorage, MemoryItem, MemoryRole};
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_offload_threshold() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path().to_path_buf());
        let config = OffloaderConfig { threshold: 10, preview_len: 5 };
        let offloader = ToolOffloader::new(storage, config);

        let mut context = Context {
            items: vec![
                MemoryItem {
                    role: MemoryRole::Tool,
                    content: "short".to_string(),
                    timestamp: 1625097600,
                    metadata: serde_json::json!({}),
                },
                MemoryItem {
                    role: MemoryRole::Tool,
                    content: "this is a very long tool result that exceeds threshold".to_string(),
                    timestamp: 1625097601,
                    metadata: serde_json::json!({}),
                },
            ],
        };

        offloader.process(&mut context).await.unwrap();

        // Verify the short message is untouched
        assert_eq!(context.items[0].content, "short");

        // Verify the long message is stubbed
        assert!(context.items[1].content.contains("Offloaded"));
        assert!(context.items[1].content.contains("this ")); // Preview check
        
        // Verify metadata was updated
        assert_eq!(context.items[1].metadata["offloaded"], true);
        
        // Verify physical file exists
        let storage_id = context.items[1].metadata["storage_id"].as_str().unwrap();
        assert!(dir.path().join(storage_id).exists());
    }

    #[tokio::test]
    async fn test_cas_deduplication() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path().to_path_buf());
        let config = OffloaderConfig { threshold: 5, preview_len: 5 };
        let offloader = ToolOffloader::new(storage, config);

        let content = "identical large content";
        let mut context = Context {
            items: vec![
                MemoryItem {
                    role: MemoryRole::Tool,
                    content: content.to_string(),
                    timestamp: 0,
                    metadata: serde_json::json!({}),
                },
                MemoryItem {
                    role: MemoryRole::Tool,
                    content: content.to_string(),
                    timestamp: 1,
                    metadata: serde_json::json!({}),
                },
            ],
        };

        offloader.process(&mut context).await.unwrap();

        let hash_1 = &context.items[0].metadata["storage_id"];
        let hash_2 = &context.items[1].metadata["storage_id"];
        
        // Verify hashes are identical (CAS)
        assert_eq!(hash_1, hash_2);
    }
}
