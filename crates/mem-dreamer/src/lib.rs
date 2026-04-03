use mem_core::{MemoryLayer, Context, LlmClient, StorageBackend, MindPalaceConfig};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use fs4::FileExt;
use std::fs::File;
use std::path::PathBuf;
use tokio::task::JoinHandle;

/// A background worker for deep-review and consolidation of session history.
///
/// The DreamWorker performs "offline" analysis of past conversation sessions, 
/// synthesizing high-level structural knowledge and persisting it as 
/// consolidated markdown files in the `knowledge/` directory.
pub struct DreamWorker<S: StorageBackend> {
    /// Client for model calls during memory synthesis.
    pub llm: Arc<dyn LlmClient>,
    /// Backend for retrieving historical sessions and storing synthesis results.
    pub storage: S,
    /// System-wide configuration for thresholds and limits.
    pub config: MindPalaceConfig,
    /// Path to a global lock file to prevent concurrent consolidation attempts.
    pub lock_path: PathBuf,
}

impl<S: StorageBackend> DreamWorker<S> {
    /// Initializes a new DreamWorker with the specified backend and configuration.
    pub fn new(llm: Arc<dyn LlmClient>, storage: S, config: MindPalaceConfig, lock_path: PathBuf) -> Self {
        Self { llm, storage, config, lock_path }
    }

    /// Executes a single consolidation cycle across all historical sessions.
    ///
    /// This method performs exclusive file-locking and summarizes session 
    /// patterns into durable knowledge markdown files.
    pub async fn run_dream_cycle(&self) -> anyhow::Result<()> {
        let file = File::create(&self.lock_path)?;
        if file.try_lock_exclusive().is_err() {
            tracing::info!("Consolidation lock active. Skipping cycle.");
            return Ok(());
        }

        // Iterative review of conversation history files.
        let sessions = self.storage.list("sessions/").await?;
        for session_name in sessions {
            if !session_name.ends_with(".json") { continue; }
            let session_id = format!("sessions/{}", session_name);
            let data = self.storage.retrieve(&session_id).await?;
            let context: Context = serde_json::from_slice(&data)?;

            let mut history = String::new();
            for item in &context.items {
                history.push_str(&format!("{:?}: {}\n", item.role, item.content));
            }

            let prompt = format!(
                "CONSOLIDATE MEMORY: Synthesize high-level structural knowledge from the session history below. \
                Look for goal patterns, technical constraints, and evolving facts. \
                Return the results as a list of bullet points.\n\n\
                HISTORY:\n{}",
                history
            );

            let synthesis = self.llm.completion(&prompt).await?;
            
            // Persist synthesis results (stored as narrative/knowledge expansion) for Layer 6.
            let synthesis_path = format!("knowledge/synthesis_{}.md", session_name);
            self.storage.store(&synthesis_path, synthesis.as_bytes()).await?;
            
            tracing::info!("Consolidated session knowledge into: {}", synthesis_path);
        }

        file.unlock()?;
        Ok(())
    }
}

/// A background scheduler that triggers consolidation cycles based on user idleness.
pub struct DreamScheduler<S: StorageBackend> {
    /// The consolidation worker instance.
    pub worker: Arc<DreamWorker<S>>,
    /// Shared atomic timestamp of the last recorded user interaction.
    pub last_activity: Arc<AtomicU64>,
    /// Background task handle.
    pub handle: Option<JoinHandle<()>>,
}

impl<S: StorageBackend + 'static> DreamScheduler<S> {
    /// Initializes a new DreamScheduler with the specified worker.
    pub fn new(worker: Arc<DreamWorker<S>>) -> Self {
        Self {
            worker,
            last_activity: Arc::new(AtomicU64::new(Self::now())),
            handle: None,
        }
    }

    /// Returns the current Unix timestamp in seconds.
    fn now() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    }

    /// Updates the `last_activity` timestamp to reflect new user interaction.
    pub fn record_activity(&self) {
        self.last_activity.store(Self::now(), Ordering::Relaxed);
    }

    /// Spawns a long-running background task that polls for idle thresholds.
    pub fn start(&mut self) {
        let worker = Arc::clone(&self.worker);
        let last_activity = Arc::clone(&self.last_activity);
        let idle_threshold = worker.config.idle_threshold_mins * 60;

        let handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                let now = Self::now();
                let last = last_activity.load(Ordering::Relaxed);
                let idle_time = now.saturating_sub(last);

                // Trigger dream cycle when idleness exceeds the threshold.
                if idle_time > idle_threshold {
                    tracing::info!("Triggering background memory consolidation after {}s idleness", idle_time);
                    if let Err(e) = worker.run_dream_cycle().await {
                        tracing::error!("Consolidation cycle failed: {:?}", e);
                    }
                    // Reset idle activity after successful consolidation cycle.
                    last_activity.store(Self::now(), Ordering::Relaxed);
                }
            }
        });

        self.handle = Some(handle);
    }

    /// Aborts the current background task.
    pub fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

#[async_trait]
impl<S: StorageBackend> MemoryLayer for DreamWorker<S> {
    fn name(&self) -> &str { "DreamWorker" }
    /// DreamWorker process does nothing in the active context pipeline; it runs offline.
    async fn process(&self, _context: &mut Context) -> anyhow::Result<()> { Ok(()) }
    /// Lowest priority ensures this layer doesn't interfere with real-time operations.
    fn priority(&self) -> u32 { 6 }
}
