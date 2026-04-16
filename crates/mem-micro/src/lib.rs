use async_trait::async_trait;
use mem_core::{Context, MemoryLayer, MemoryRole, MindPalaceConfig, RelevanceAnalyzer};
use std::sync::Arc;

/// Strategy for determining how a memory item's Time-To-Live (TTL) decays over time.
pub enum TTLDecayStrategy {
    /// Linear decay based on a fixed slope (age * slope).
    Linear { slope: f32 },
    /// Exponential decay based on a specified half-life in seconds.
    Exponential { half_life: u64 },
    /// Dynamically adjusts TTL based on message role and context relevance scores.
    AdaptiveByType,
}

/// A short-term memory layer that prunes context based on age and local relevance.
///
/// Unlike naive TTL, the AdaptiveMicroCompactor uses a `RelevanceAnalyzer` to
/// provide "stickiness" to important recent messages, extending their TTL
/// even if they exceed the base age threshold.
pub struct AdaptiveMicroCompactor {
    /// System-wide configuration for thresholds and limits.
    pub config: MindPalaceConfig,
    /// Strategy to apply when calculating the effective TTL.
    pub decay_function: TTLDecayStrategy,
    /// Analyzer for determining a message's importance to the current task.
    pub relevance_analyzer: Arc<dyn RelevanceAnalyzer>,
}

impl AdaptiveMicroCompactor {
    /// Initializes a new AdaptiveMicroCompactor with the specified configuration and decay strategy.
    pub fn new(
        config: MindPalaceConfig,
        decay_function: TTLDecayStrategy,
        relevance_analyzer: Arc<dyn RelevanceAnalyzer>,
    ) -> Self {
        Self {
            config,
            decay_function,
            relevance_analyzer,
        }
    }
}

#[async_trait]
impl MemoryLayer for AdaptiveMicroCompactor {
    fn name(&self) -> &str {
        "AdaptiveMicroCompactor"
    }

    /// Prunes stale items from the context based on their age and relevance.
    ///
    /// System-level messages are always retained (immortal).
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        let now = chrono::Utc::now().timestamp() as u64;

        let mut to_remove = Vec::new();

        for (idx, item) in context.items.iter().enumerate() {
            // Preservation policy: Always keep System messages.
            if item.role == MemoryRole::System {
                continue;
            }

            let age = now.saturating_sub(item.timestamp);
            let relevance = self
                .relevance_analyzer
                .score_relevance(item, context)
                .await?;

            // Adjust TTL based on relevance and the chosen strategy.
            let effective_ttl = match &self.decay_function {
                TTLDecayStrategy::AdaptiveByType => {
                    // Relevance boost: Keep high-relevance items up to 3x longer.
                    (self.config.base_ttl_seconds as f32 * (1.0 + relevance * 2.0)) as u64
                }
                TTLDecayStrategy::Exponential { half_life } => {
                    let decay = 2_f32.powf(-(age as f32) / (*half_life as f32));
                    (self.config.base_ttl_seconds as f32 * decay * (1.0 + relevance)) as u64
                }
                TTLDecayStrategy::Linear { slope } => {
                    let decay = (1.0 - (age as f32 * slope)).max(0.0);
                    (self.config.base_ttl_seconds as f32 * decay * (1.0 + relevance)) as u64
                }
            };

            if age > effective_ttl {
                to_remove.push(idx);
            }
        }

        // Remove stale items in reverse order to preserve indexing stability.
        for idx in to_remove.into_iter().rev() {
            context.items.remove(idx);
        }

        Ok(())
    }

    /// Higher priority: Micro-compaction should run early to prune obvious noise.
    fn priority(&self) -> u32 {
        2
    }
}
