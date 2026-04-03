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
        tracing::info!("Context frozen: {}", snapshot_id);
        Ok(())
    }

    /// Forks a child context from an existing snapshot for recovery or parallel tasks.
    pub async fn fork_context(&self, snapshot_id: &str) -> anyhow::Result<Context> {
        let path = format!("snapshots/{}.json", snapshot_id);
        let data = self.storage.retrieve(&path).await?;
        let context: Context = serde_json::from_slice(&data)?;
        tracing::info!("Context forked from: {}", snapshot_id);
        Ok(context)
    }
}

#[async_trait]
impl<S: StorageBackend> MemoryLayer for AgentBridge<S> {
    fn name(&self) -> &str {
        "AgentBridge"
    }

    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        // Gap 7: Multi-agent coordination logic.
        // If the context has a 'freeze' metadata trigger, perform an automatic snapshot.
        let should_freeze = context.items.last().map_or(false, |i| {
            i.metadata["freeze_trigger"].as_str().is_some()
        });

        if should_freeze {
            let id = context.items.last().unwrap().metadata["freeze_trigger"].as_str().unwrap().to_string();
            self.freeze_context(&id, context).await?;
        }

        Ok(())
    }

    fn priority(&self) -> u32 {
        7 // Final layer (after optimization and compaction)
    }
}
