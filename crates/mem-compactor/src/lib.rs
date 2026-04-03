use mem_core::{MemoryLayer, Context, LlmClient, MemoryItem, MemoryRole, ImportanceAnalyzer, StorageBackend, MindPalaceConfig};
use async_trait::async_trait;
use std::sync::Arc;

/// A heavy-duty memory layer for structural context compression when the session exceeds capacity.
///
/// The IntelligentFullCompactor scores every context item by objective importance 
/// and summarizes less-relevant segments using a 9-point structural model. 
/// It also creates pre-compaction checkpoints for safety and disaster recovery.
pub struct IntelligentFullCompactor<S: StorageBackend> {
    /// Client for model calls during importance analysis and summarization.
    pub llm: Arc<dyn LlmClient>,
    /// Scoring service to determine which items are critical for retention.
    pub importance_analyzer: Arc<dyn ImportanceAnalyzer>,
    /// Backend for persisting pre-compaction checkpoints.
    pub storage: S,
    /// System-wide configuration for thresholds and limits.
    pub config: MindPalaceConfig,
    /// Directory for storing safety checkpoints.
    pub checkpoint_dir: String,
}

impl<S: StorageBackend> IntelligentFullCompactor<S> {
    /// Initializes a new IntelligentFullCompactor for the specified storage directory.
    pub fn new(
        llm: Arc<dyn LlmClient>,
        importance_analyzer: Arc<dyn ImportanceAnalyzer>,
        storage: S,
        config: MindPalaceConfig,
        checkpoint_dir: String,
    ) -> Self {
        Self {
            llm,
            importance_analyzer,
            storage,
            config,
            checkpoint_dir,
        }
    }

    /// Persists a full serialized snapshot of the current context for safety and forensics.
    pub async fn create_checkpoint(&self, context: &Context) -> anyhow::Result<String> {
        let timestamp = chrono::Utc::now().timestamp();
        let filename = format!("{}/checkpoint_{}.json", self.checkpoint_dir, timestamp);
        let data = serde_json::to_vec_pretty(context)?;
        self.storage.store(&filename, &data).await?;
        Ok(filename)
    }
}

/// The system prompt for 9-point structural summarization (preserving goals, progress, etc.)
pub const STRUCTURAL_SUMMARY_PROMPT: &str = r#"
Apply 9-point structural summarization to the following conversation history. 
Preserve the core technical intent, constraints, and progress.

1. GOAL: Original user objective.
2. CONTEXT: Environmental facts.
3. TOOLS: Successfully used tools.
4. ERRORS: Technical failures that inform the current approach.
5. PROGRESS: Percentage towards goal.
6. PENDING: Immediate next steps.
7. CONSTANTS: Immutable facts/constraints.
8. PREFERENCES: User formatting/style preferences.
9. NEXT: Very next action.

HISTORY:
{history}
"#;

#[async_trait]
impl<S: StorageBackend> MemoryLayer for IntelligentFullCompactor<S> {
    fn name(&self) -> &str {
        "IntelligentFullCompactor"
    }

    /// Executes the full compaction cycle, including checkpointing and importance-based pruning.
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        if context.items.len() < self.config.max_context_items {
            return Ok(());
        }

        // 1. Safety Checkpoint: Create full backup before heavy transformation.
        let checkpoint_id = self.create_checkpoint(context).await?;
        tracing::info!("Created pre-compaction checkpoint: {}", checkpoint_id);

        // 2. Objective Scrutiny: Determine which items must remain in context.
        let mut scored_items: Vec<(usize, MemoryItem, f32)> = Vec::new();
        for (idx, item) in context.items.iter().enumerate() {
            let score = self.importance_analyzer.score_importance(item, context).await?;
            scored_items.push((idx, item.clone(), score));
        }

        // 3. Keep high-importance items regardless of their position in time.
        // We keep the top 1/3 of items by importance score.
        scored_items.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        let keep_count = (scored_items.len() / 3).max(1);
        let high_importance: Vec<_> = scored_items.iter().take(keep_count).map(|(_, item, _)| item.clone()).collect();
        let to_summarize: Vec<_> = scored_items.iter().skip(keep_count).map(|(_, item, _)| item.clone()).collect();

        // 4. Summarize less important items into a unified structural state.
        let mut history = String::new();
        for item in &to_summarize {
            history.push_str(&format!("{:?}: {}\n", item.role, item.content));
        }

        let prompt = STRUCTURAL_SUMMARY_PROMPT.replace("{history}", &history);
        let summary_text = self.llm.completion(&prompt).await?;

        let summary_item = MemoryItem {
            role: MemoryRole::System,
            content: format!("### COMPACTED SESSION STATE ###\n\n{}", summary_text),
            timestamp: chrono::Utc::now().timestamp() as u64,
            metadata: serde_json::json!({
                "compaction": true,
                "importance_filtered": true,
                "layer": 4,
                "summarized_count": to_summarize.len(),
                "checkpoint": checkpoint_id
            }),
        };

        // 5. Final Reconstruction: Summary + preserved high-importance items.
        // History is reconstructed by preserving the relative ordering of high-importance items.
        let mut finalized_items = vec![summary_item];
        finalized_items.extend(high_importance);
        finalized_items.sort_by_key(|i| i.timestamp);
        
        context.items = finalized_items;

        Ok(())
    }

    /// Lowest priority: Major compaction should occur after all other filtering and extraction.
    fn priority(&self) -> u32 {
        4
    }
}
