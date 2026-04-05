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

impl MindPalaceConfig {
    /// Validate configuration ranges
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            anyhow::bail!(
                "similarity_threshold must be in [0.0, 1.0], got {}",
                self.similarity_threshold
            );
        }
        
        if !(0.0..=1.0).contains(&self.compression_ratio) {
            anyhow::bail!(
                "compression_ratio must be in [0.0, 1.0], got {}",
                self.compression_ratio
            );
        }
        
        if self.max_context_items == 0 {
            anyhow::bail!("max_context_items must be > 0");
        }
        
        if self.max_tokens_per_dream == 0 {
            anyhow::bail!("max_tokens_per_dream must be > 0");
        }
        
        if self.default_model.is_empty() {
            anyhow::bail!("default_model cannot be empty");
        }
        
        Ok(())
    }
    
    /// Load from environment variables, falling back to defaults if not present
    pub fn from_env() -> Self {
        let config = Self {
            similarity_threshold: std::env::var("MINDPALACE_SIMILARITY_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.85),
            
            compression_ratio: std::env::var("MINDPALACE_COMPRESSION_RATIO")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.6),
            
            max_context_items: std::env::var("MINDPALACE_MAX_CONTEXT_ITEMS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100),
            
            base_ttl_seconds: std::env::var("MINDPALACE_BASE_TTL_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600),
            
            idle_threshold_mins: std::env::var("MINDPALACE_IDLE_THRESHOLD_MINS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(45),
            
            summary_interval: std::env::var("MINDPALACE_SUMMARY_INTERVAL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(15),
            
            default_model: std::env::var("MINDPALACE_DEFAULT_MODEL")
                .unwrap_or_else(|_| "gpt-4-turbo".to_string()),
            
            max_tokens_per_dream: std::env::var("MINDPALACE_MAX_TOKENS_PER_DREAM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50_000),
            
            retention_sessions: std::env::var("MINDPALACE_RETENTION_SESSIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
        };
        
        if let Err(e) = config.validate() {
            tracing::error!("Configuration validation failed: {}", e);
        }
        
        config
    }
}

impl Default for MindPalaceConfig {
    fn default() -> Self {
        Self::from_env()
    }
}
