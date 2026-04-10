use mem_core::{MemoryLayer, Context, MemoryMetrics, TokenCounter, MindPalaceConfig, MemoryRole};
use std::sync::Arc;
use std::time::Instant;

/// The central orchestrator for the MindPalace memory system.
///
/// The Brain manages a pipeline of `MemoryLayer` implementations and applies them 
/// to the agent's conversation context in a prioritized sequence.
pub struct Brain {
    /// Ordered list of active memory layers.
    pub layers: Vec<Arc<dyn MemoryLayer>>,
    /// Optional collection of Prometheus metrics for system health.
    pub metrics: Option<MemoryMetrics>,
    /// Optional counter for precise token calculation across model providers.
    pub token_counter: Option<Arc<dyn TokenCounter>>,
    /// System-wide configuration for thresholds and limits.
    pub config: MindPalaceConfig,
}

impl Brain {
    /// Initializes a new Brain instance with the specified configuration.
    pub fn new(config: MindPalaceConfig, metrics: Option<MemoryMetrics>, token_counter: Option<Arc<dyn TokenCounter>>) -> Self {
        Self { 
            layers: Vec::new(),
            metrics,
            token_counter,
            config,
        }
    }

    /// Registers a new memory layer and re-sorts the pipeline by priority.
    pub fn add_layer(&mut self, layer: Arc<dyn MemoryLayer>) {
        self.layers.push(layer);
        self.layers.sort_by_key(|l| l.priority());
    }

    /// Executes the full memory optimization pipeline on the provided context.
    ///
    /// This method sequentially runs all registered layers, tracks their performance 
    /// (latency), and updates system-wide metrics including token usage and 
    /// compression ratios.
    pub async fn optimize(&self, context: &mut Context) -> anyhow::Result<()> {
        let initial_size = serde_json::to_vec(context)?.len();
        let initial_items = context.items.len();
        
        // Calculate baseline token usage if a counter is available.
        let initial_tokens = if let Some(ref counter) = self.token_counter {
            let total: usize = context.items.iter().map(|i| counter.count_tokens(&i.content)).sum();
            Some(total)
        } else {
            None
        };

        // Execute prioritized layers.
        for layer in &self.layers {
            let start = Instant::now();
            if let Err(e) = layer.process(context).await {
                tracing::error!("Memory layer '{}' failed: {:?}", layer.name(), e);
                return Err(e);
            }
            let duration = start.elapsed();

            // Record layer latency in Prometheus.
            if let Some(metrics) = &self.metrics {
                metrics.layer_latency.observe(duration.as_secs_f64());
            }
        }

        let final_size = serde_json::to_vec(context)?.len();
        let final_items = context.items.len();

        // Update system health metrics.
        if let Some(metrics) = &self.metrics {
            metrics.context_size_bytes.set(final_size as f64);
            metrics.item_count.set(final_items as f64);
            
            // Calculate final segment token usage and update cumulative metrics.
            if let (Some(ref counter), Some(start_tokens)) = (&self.token_counter, initial_tokens) {
                let final_tokens: usize = context.items.iter().map(|i| counter.count_tokens(&i.content)).sum();
                metrics.total_tokens_processed.add(final_tokens as f64);
                
                tracing::info!(
                    "Token usage: {} -> {} ({}% compression)",
                    start_tokens,
                    final_tokens,
                    if start_tokens > 0 { (final_tokens as f32 / start_tokens as f32) * 100.0 } else { 100.0 }
                );
            }

            if initial_size > 0 {
                metrics.compression_ratio.set((final_size as f64 / initial_size as f64) * 100.0);
            }
        }

        // 5. Final Budget Enforcement (Hard Limit Protection).
        // Ensure that context never exceeds max_context_items, even if layers fail to compress.
        if context.items.len() > self.config.max_context_items {
            tracing::warn!(
                "Context budget exceeded: {} > {}. Performing emergency pruning.",
                context.items.len(),
                self.config.max_context_items
            );
            
            // 1. Identify System Prompt (must be kept)
            let mut final_items = Vec::new();
            if let Some(system_msg) = context.items.iter().find(|i| i.role == MemoryRole::System).cloned() {
                final_items.push(system_msg);
            }
            
            // 2. Calculate initial retention target
            let target_retain = self.config.max_context_items.saturating_sub(final_items.len());
            let mut start_idx = context.items.len().saturating_sub(target_retain);

            // 3. OpenAI Sequence Safety: Ensure we don't start with an orphaned Tool message.
            // If the message at start_idx is a 'Tool' role, we must back up to include the 
            // preceding 'Assistant' message that initiated the call.
            while start_idx > 0 && start_idx < context.items.len() {
                let current_role = &context.items[start_idx].role;
                if current_role == &MemoryRole::Tool {

                    // Back up to find the assistant call or a non-tool message
                    start_idx -= 1;
                } else if current_role == &MemoryRole::Assistant {
                    // If this is an assistant message, check if it has tool calls.
                    // If it does, we found the sequence start.
                    // If it doesn't but is followed by a tool, it's the start of a sequence anyway.
                    let next_is_tool = start_idx + 1 < context.items.len() && context.items[start_idx + 1].role == MemoryRole::Tool;
                    if next_is_tool {
                        // Keep this assistant message to satisfy the tool result sequence.
                        break;
                    }
                    // Not followed by a tool, so this is a safe split point.
                    break;
                } else {
                    // Found a non-tool, non-assistant message (User or System). Safe split point.
                    break;
                }

            }
            
            // Finalize items
            final_items.extend(context.items.drain(start_idx..));
            context.items = final_items;
        }


        tracing::info!(
            "Memory optimized: {} items -> {} items ({}% compression)",
            initial_items,
            context.items.len(),
            if initial_size > 0 { (final_size as f32 / initial_size as f32) * 100.0 } else { 100.0 }
        );

        Ok(())
    }
}

impl Default for Brain {
    /// Creates a default Brain with standard configuration.
    fn default() -> Self {
        Self::new(MindPalaceConfig::default(), None, None)
    }
}
