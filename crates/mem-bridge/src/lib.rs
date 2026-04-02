use mem_core::{MemoryLayer, Context, StorageBackend};
use async_trait::async_trait;

pub struct AgentBridge<S: StorageBackend> {
    pub storage: S,
}

impl<S: StorageBackend> AgentBridge<S> {
    pub fn new(storage: S) -> Self {
        Self { storage }
    }

    /// Freezes the current context into a persistent snapshot.
    /// This ensures child agents inherit a byte-identical prefix (Prompt Cache efficient).
    pub async fn freeze_context(&self, snapshot_id: &str, context: &Context) -> anyhow::Result<()> {
        let path = format!("snapshots/{}.json", snapshot_id);
        let data = serde_json::to_vec_pretty(context)?;
        self.storage.store(&path, &data).await?;
        Ok(())
    }

    /// Forks a child context from an existing snapshot for recovery or parallel tasks.
    pub async fn fork_context(&self, snapshot_id: &str) -> anyhow::Result<Context> {
        let path = format!("snapshots/{}.json", snapshot_id);
        let data = self.storage.retrieve(&path).await?;
        let context: Context = serde_json::from_slice(&data)?;
        Ok(context)
    }
}

#[async_trait]
impl<S: StorageBackend> MemoryLayer for AgentBridge<S> {
    fn name(&self) -> &str {
        "AgentBridge"
    }

    async fn process(&self, _context: &mut Context) -> anyhow::Result<()> {
        // Layer 7 handles cross-agent coordination logic.
        Ok(())
    }

    fn priority(&self) -> u32 {
        7
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mem_core::{FileStorage, MemoryItem, MemoryRole};
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_freeze_and_fork_recovery() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path().to_path_buf());
        let bridge = AgentBridge::new(storage);

        let context = Context {
            items: vec![MemoryItem {
                role: MemoryRole::User,
                content: "Freeze me".to_string(),
                timestamp: 123,
                metadata: serde_json::json!({}),
            }],
        };

        // 1. Freeze
        bridge.freeze_context("parent_001", &context).await.unwrap();
        
        // 2. Verify file exists in snapshots
        assert!(dir.path().join("snapshots/parent_001.json").exists());

        // 3. Fork (Recovery)
        let forked = bridge.fork_context("parent_001").await.unwrap();
        assert_eq!(forked.items[0].content, "Freeze me");
    }
}
