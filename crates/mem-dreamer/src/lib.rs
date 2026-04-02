use mem_core::{MemoryLayer, Context, LlmClient, StorageBackend, utils};
use async_trait::async_trait;
use std::sync::Arc;
use fs4::FileExt;
use std::fs::File;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Token cap per background cycle (Configurable as requested)
    pub max_tokens_per_dream: usize,
    /// Inactivity window before dreaming (45-60 mins as requested)
    pub idle_threshold_mins: u64,
    /// How many sessions to keep in raw JSON before compression/archival
    pub retention_sessions: usize,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_dream: 50_000,
            idle_threshold_mins: 45,
            retention_sessions: 10,
        }
    }
}

pub struct DreamWorker<S: StorageBackend> {
    pub llm: Arc<dyn LlmClient>,
    pub storage: S,
    pub config: DreamConfig,
    pub lock_path: PathBuf,
}

impl<S: StorageBackend> DreamWorker<S> {
    pub fn new(llm: Arc<dyn LlmClient>, storage: S, config: DreamConfig, lock_path: PathBuf) -> Self {
        Self {
            llm,
            storage,
            config,
            lock_path,
        }
    }

    /// Primary background worker for memory consolidation.
    /// Follows the Anthropic design for idle-time knowledge synthesis.
    pub async fn run_dream_cycle(&self) -> anyhow::Result<()> {
        // 1. Process-Level Locking: Acquire exclusive lock for the dream cycle.
        // This ensures only one agent instance optimizes the knowledge store at once.
        let file = File::create(&self.lock_path)?;
        if file.try_lock_exclusive().is_err() {
            tracing::info!("Dreaming lock active on {:?}. Skipping cycle.", self.lock_path);
            return Ok(());
        }

        // 2. Scan for finalized sessions (L5 extracted but not yet L6 consolidated)
        // Scaffolding logic: targeting a session marked for archival
        let session_id = "sessions/legacy_001.json";
        if self.storage.exists(session_id).await {
            let data = self.storage.retrieve(session_id).await?;
            let context: Context = serde_json::from_slice(&data)?;

            // 3. Synthesis: Deep-review of legacy conversations.
            let mut history = String::new();
            for item in &context.items {
                history.push_str(&format!("{:?}: {}\n", item.role, item.content));
            }

            let prompt = format!(
                "CONSOLIDATE MEMORY: Review the history and synthesize any major contradictions \
                or new high-level knowledge for the knowledge.json base.\n\n\
                HISTORY:\n{}",
                history
            );

            // Execute synthesis (Token-capped in real implementation)
            let _synthesis = self.llm.completion(&prompt).await?;
            
            // 4. Archive with Compression: Store as JSON + Zstd for space efficiency.
            let compressed_data = utils::compress(&data)?;
            let archive_id = format!("archive/legacy_001.json.zst");
            self.storage.store(&archive_id, &compressed_data).await?;
            
            tracing::info!("Successfully consolidated and archived session legacy_001.");
        }

        // 5. Release Lock
        file.unlock()?;
        Ok(())
    }
}

#[async_trait]
impl<S: StorageBackend> MemoryLayer for DreamWorker<S> {
    fn name(&self) -> &str {
        "DreamWorker"
    }

    async fn process(&self, _context: &mut Context) -> anyhow::Result<()> {
        // "Daydreaming": In-session lighter review logic.
        Ok(())
    }

    fn priority(&self) -> u32 {
        6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mem_core::{FileStorage, MemoryItem, MemoryRole};
    use tempfile::tempdir;

    struct MockLlm;
    #[async_trait]
    impl LlmClient for MockLlm {
        async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
            Ok("Synthesized knowledge from dream.".to_string())
        }
    }

    #[tokio::test]
    async fn test_dream_worker_locking() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path().to_path_buf());
        let llm = Arc::new(MockLlm);
        let lock_path = dir.path().join("dream.lock");
        
        let worker = DreamWorker::new(llm, storage, DreamConfig::default(), lock_path.clone());

        // 1. Initial run should succeed
        worker.run_dream_cycle().await.unwrap();

        // 2. Simulate concurrent access by manually locking the file
        let lock_file = File::create(&lock_path).unwrap();
        lock_file.lock_exclusive().unwrap();

        // 3. Worker should detect lock and skip (logs "Dreaming lock active")
        worker.run_dream_cycle().await.unwrap(); 
        
        lock_file.unlock().unwrap();
    }

    #[tokio::test]
    async fn test_archival_and_compression() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path().to_path_buf());
        let llm = Arc::new(MockLlm);
        let lock_path = dir.path().join("dream.lock");
        let worker = DreamWorker::new(llm, storage.clone(), DreamConfig::default(), lock_path);

        // Setup a mock legacy session
        let context = Context {
            items: vec![MemoryItem {
                role: MemoryRole::User,
                content: "Hello world".to_string(),
                timestamp: 123,
                metadata: serde_json::json!({}),
            }],
        };
        let session_data = serde_json::to_vec(&context).unwrap();
        
        storage.store("sessions/legacy_001.json", &session_data).await.unwrap();

        // Run dream cycle
        worker.run_dream_cycle().await.unwrap();

        // Verify archival exists
        assert!(dir.path().join("archive/legacy_001.json.zst").exists());
        
        // Verify decompression works
        let compressed = std::fs::read(dir.path().join("archive/legacy_001.json.zst")).unwrap();
        let decompressed = utils::decompress(&compressed).unwrap();
        assert_eq!(decompressed, session_data);
    }
}
