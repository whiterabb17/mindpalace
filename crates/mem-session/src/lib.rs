use async_trait::async_trait;
use mem_core::{
    Context, LlmClient, MemoryItem, MemoryLayer, MemoryRole, MindPalaceConfig, StorageBackend,
};
use std::sync::Arc;

/// A memory layer that compresses historical conversation segments into narrative summaries.
///
/// The SessionSummarizer prevents short-term memory bloat by periodically iterating
/// over the current context, generating a high-fidelity summary of older messages,
/// and replacing them with a single summary item.
pub struct SessionSummarizer<S: StorageBackend> {
    /// Client for model calls during summary generation.
    pub llm: Arc<dyn LlmClient>,
    /// Backend for persisting summary markdown files.
    pub storage: S,
    /// System-wide configuration for thresholds and limits.
    pub config: MindPalaceConfig,
    /// If true, validates that the summary retains all critical facts before replacement.
    pub validation_mode: bool,
    /// Directory for storing narrative markdown files.
    pub narrative_dir: String,
}

impl<S: StorageBackend> SessionSummarizer<S> {
    /// Initializes a new SessionSummarizer for the specified storage directory.
    pub fn new(
        llm: Arc<dyn LlmClient>,
        storage: S,
        config: MindPalaceConfig,
        narrative_dir: String,
        validation_mode: bool,
    ) -> Self {
        Self {
            llm,
            storage,
            config,
            narrative_dir,
            validation_mode,
        }
    }

    /// Validates the generated summary against the original context for fidelity.
    pub async fn validate_summary_fidelity(
        &self,
        summary: &str,
        context: &Context,
    ) -> anyhow::Result<bool> {
        let mut history = String::new();
        for item in &context.items {
            history.push_str(&format!("{:?}: {}\n", item.role, item.content));
        }
        let prompt = format!("Verify if summary retains all critical facts without hallucinations. Return 'PASS' or 'FAIL: [reason]'.\n\nHISTORY:\n{}\n\nSUMMARY:\n{}", history, summary);
        let res = self.llm.completion(&prompt).await?;
        Ok(res.contains("PASS"))
    }

    /// Generates a technical markdown summary for the current conversation history.
    pub async fn generate_summary(&self, context: &Context) -> anyhow::Result<String> {
        let mut history = String::new();
        for item in &context.items {
            history.push_str(&format!("{:?}: {}\n", item.role, item.content));
        }

        let prompt = format!(
            "Provide a comprehensive markdown summary of the progress and conversation state so far. \
            Be technical and precise. This summary will replace the earlier messages to save context space.\n\n\
            HISTORY:\n{}",
            history
        );

        self.llm.completion(&prompt).await
    }
}

#[async_trait]
impl<S: StorageBackend> MemoryLayer for SessionSummarizer<S> {
    fn name(&self) -> &str {
        "SessionSummarizer"
    }

    /// Executes the summarization pipeline, including fidelity checks and markdown persistence.
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        if context.items.len() < self.config.summary_interval {
            return Ok(());
        }

        // 1. Generate summary.
        let summary = self.generate_summary(context).await?;

        // 2. Perform fidelity validation if enabled.
        if self.validation_mode && !self.validate_summary_fidelity(&summary, context).await? {
            tracing::warn!("Summary fidelity check FAILED. Retaining original context.");
            return Ok(());
        }

        // 3. Persist narrative to markdown (Gap 3 & Improvement 3).
        let timestamp = chrono::Utc::now().timestamp();
        let filename = format!("{}/narrative_{}.md", self.narrative_dir, timestamp);
        self.storage.store(&filename, summary.as_bytes()).await?;

        // 4. Comprehensive contextual pruning (Gap 3).
        // Calculation ensures that latest messages are retained while the history is condensed.
        let cutoff_idx = (context.items.len() as f32 * self.config.compression_ratio) as usize;
        let original_count = context.items.len();

        let summary_item = MemoryItem {
            role: MemoryRole::System,
            content: format!("### SESSION NARRATIVE SUMMARY ###\n\n{}", summary),
            timestamp: timestamp as u64,
            metadata: serde_json::json!({
                "type": "session_summary",
                "original_item_count": original_count,
                "compressed_up_to": cutoff_idx,
                "narrative_file": filename
            }),
        };

        // Efficient context compression: Summary + Recent history.
        let mut new_items = vec![summary_item];
        new_items.extend(context.items.drain(cutoff_idx..));
        context.items = new_items;

        Ok(())
    }

    /// Medium priority: summaries should run before micro-compaction but after extractor layers.
    fn priority(&self) -> u32 {
        3
    }
}
