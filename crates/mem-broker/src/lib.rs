use mem_core::{FactNode, FactScope, StorageBackend};
use fs4::FileExt;
use std::fs::File;
use std::path::PathBuf;

pub mod tools;

/// A central broker for broadcasting and synchronizing multi-agent knowledge.
///
/// The FactBroker facilitates "Collective Learning" by allowing agents to publish 
/// high-confidence technical or objective facts to a shared pool and subscribe 
/// to updates from other agents in the fleet.
pub struct FactBroker<S: StorageBackend> {
    /// Backend for shared knowledge pools (e.g., a shared network drive or S3).
    pub shared_storage: S,
    /// Path to the global lock file for atomic shared pool updates.
    pub lock_path: PathBuf,
}

impl<S: StorageBackend> FactBroker<S> {
    /// Initializes a new FactBroker with the specified shared storage and lock path.
    pub fn new(shared_storage: S, lock_path: PathBuf) -> Self {
        Self { shared_storage, lock_path }
    }

    /// Publishes high-confidence global or project-scoped facts to the shared pool.
    ///
    /// Facts are filtered by confidence (>= 0.9) and scope before publication.
    pub async fn publish_facts(&self, facts: Vec<FactNode>) -> anyhow::Result<usize> {
        let to_publish: Vec<_> = facts.into_iter()
            .filter(|f| (f.scope == FactScope::Global || f.scope == FactScope::Project) && f.confidence >= 0.9)
            .collect();
        
        if to_publish.is_empty() { return Ok(0); }

        let file = File::create(&self.lock_path)?;
        file.lock_exclusive()?; // Ensure atomic multi-agent access

        let count = to_publish.len();
        for fact in to_publish {
            let fact_id = format!("shared_kb/{}.json", fact.id);
            let data = serde_json::to_vec_pretty(&fact)?;
            self.shared_storage.store(&fact_id, &data).await?;
        }

        file.unlock()?;
        Ok(count)
    }

    /// Retrieves all available facts from the shared pool for local integration.
    pub async fn pull_shared_knowledge(&self) -> anyhow::Result<Vec<FactNode>> {
        let shared_files = self.shared_storage.list("shared_kb/").await?;
        let mut shared_facts = Vec::new();

        for file_id in shared_files {
            if !file_id.ends_with(".json") { continue; }
            let full_id = format!("shared_kb/{}", file_id);
            let data = self.shared_storage.retrieve(&full_id).await?;
            if data.is_empty() { continue; }
            let fact: FactNode = match serde_json::from_slice(&data) {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!("Failed to parse shared fact {}: {}. Skipping.", full_id, e);
                    continue;
                }
            };
            shared_facts.push(fact);
        }

        Ok(shared_facts)
    }
}
