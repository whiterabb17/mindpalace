use mem_core::{MemoryLayer, Context, MemoryMetrics, TokenCounter};
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
}

impl Brain {
    /// Initializes a new Brain instance with optional metrics and token counting.
    pub fn new(metrics: Option<MemoryMetrics>, token_counter: Option<Arc<dyn TokenCounter>>) -> Self {
        Self { 
            layers: Vec::new(),
            metrics,
            token_counter,
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

        tracing::info!(
            "Memory optimized: {} items -> {} items ({}% compression)",
            initial_items,
            final_items,
            if initial_size > 0 { (final_size as f32 / initial_size as f32) * 100.0 } else { 100.0 }
        );

        Ok(())
    }
}

impl Default for Brain {
    /// Creates a default Brain without metrics or token counting.
    fn default() -> Self {
        Self::new(None, None)
    }
}
