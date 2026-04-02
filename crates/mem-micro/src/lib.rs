use mem_core::{MemoryLayer, Context, MemoryRole};
use async_trait::async_trait;

pub struct MicroCompactor {
    /// Maximum character length before aggressive pruning (Skeletonization)
    pub max_chars: usize,
    /// Time-To-Live for non-system messages in seconds (default: 3600)
    pub ttl_seconds: u64,
}

impl Default for MicroCompactor {
    fn default() -> Self {
        Self { 
            max_chars: 20_000,
            ttl_seconds: 3600, 
        }
    }
}

#[async_trait]
impl MemoryLayer for MicroCompactor {
    fn name(&self) -> &str {
        "MicroCompactor"
    }

    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        let now = chrono::Utc::now().timestamp() as u64;

        // TTL-based Eviction: Remove old non-system messages
        context.items.retain(|item| {
            match item.role {
                MemoryRole::System => true, // System messages are immortal
                _ => (now - item.timestamp) < self.ttl_seconds,
            }
        });

        // Whitespace & Metadata Skeletonization (Optional/Future: could be added here)
        for item in &mut context.items {
            if item.content.len() > self.max_chars {
                // Aggressive skeletonization logic would go here
            }
        }

        Ok(())
    }

    fn priority(&self) -> u32 {
        2
    }
}
