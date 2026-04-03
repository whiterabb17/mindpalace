use serde::{Deserialize, Serialize};

/// Centralized configuration for the MindPalace memory ecosystem.
///
/// This structure holds all operational thresholds, ratios, and limits 
/// required by the multi-layered memory pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MindPalaceConfig {
    /// Minimum semantic similarity (0.0 to 1.0) for a fact to be considered a duplicate.
    pub similarity_threshold: f32,
    /// Target percentage of context to compress during session summarization (0.0 to 1.0).
    pub compression_ratio: f32,
    /// Maximum number of items the short-term context is allowed to hold (Budget).
    pub max_context_items: usize,
    /// Time (seconds) until a non-system message is eligible for micro-compaction.
    pub base_ttl_seconds: u64,
    /// Idle time (minutes) before a background dream cycle is triggered.
    pub idle_threshold_mins: u64,
    /// Interval of messages before session summarization is triggered.
    pub summary_interval: usize,
    /// Default model name for LLM operations.
    pub default_model: String,
    /// Maximum number of tokens to process in a single dream cycle.
    pub max_tokens_per_dream: usize,
    /// Number of sessions to retain during background maintenance.
    pub retention_sessions: usize,
}

impl Default for MindPalaceConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            compression_ratio: 0.6,
            max_context_items: 100,
            base_ttl_seconds: 3600,
            idle_threshold_mins: 45,
            summary_interval: 15,
            default_model: "gpt-4-turbo".to_string(),
            max_tokens_per_dream: 50_000,
            retention_sessions: 10,
        }
    }
}
