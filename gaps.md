CRITICAL GAPS & IMPROVEMENTS NEEDED
Add a TokenCounter trait and implement it for all model providers.
Layer 3 must produce persistent markdown files.
Add checkpointing to layer 4 before compaction
Implement semantic deduplication for layer 5
Persist synthesis results in layer 6 
Document all public APIs
Add metrics/observability model

GAP 1: Semantic Deduplication is Insufficient
Problem:

Current Implementation (line 70, mem-extractor/src/lib.rs):

Rust
if !kb.facts.iter().any(|f| f.content.to_lowercase() == fact.content.to_lowercase())
Only uses exact string matching with case-insensitivity.

Issue: The same fact expressed differently won't be detected:

"The user prefers Rust" vs "Rust is the user's language of choice"
"Database schema v1.2.3" vs "DB schema version 1.2.3"
These are semantically identical but textually different.
Impact: Knowledge base bloat, redundant storage, LLM re-processing of same information.

Recommendation:

Rust
// Implement semantic similarity checking
pub async fn semantic_deduplication(
    &self, 
    new_fact: &Fact, 
    existing_facts: &[Fact]
) -> anyhow::Result<bool> {
    for existing in existing_facts {
        // Use embedding-based similarity (e.g., OpenAI embeddings, Ollama)
        let similarity = self.compute_embedding_distance(
            &new_fact.content, 
            &existing.content
        ).await?;
        
        // If > 0.85 similarity, consider duplicate
        if similarity > 0.85 {
            return Ok(true);
        }
    }
    Ok(false)
}
GAP 2: Fact Extraction Lacks Temporal Context & Relationships
Problem:

Current Implementation (mem-extractor):

Extracts facts with only category, content, confidence, timestamp
No graph relationships between facts
No versioning or evolution tracking
Missing:

Fact dependencies ("Fact A implies Fact B")
Temporal validity windows ("true until user changes project")
Superseded facts ("old preference replaced by new one")
Fact provenance ("extracted from which session/turn?")
Example Problem:

Code
Session 1: "User prefers Python"  → Stored as Fact
Session 5: "User switched to Rust" → New Fact added
Session 10: "Now focusing on Kotlin" → Another new Fact

Result: 3 conflicting "user language preference" facts
No way to know which is current or how they relate
Recommendation:

Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactNode {
    pub id: String,           // UUID
    pub content: String,
    pub category: String,
    pub confidence: f32,
    pub timestamp: u64,
    pub version: u32,         // NEW: Track evolution
    pub superseded_by: Option<String>,  // NEW: Point to newer fact
    pub dependencies: Vec<String>,      // NEW: Related facts
    pub valid_until: Option<u64>,       // NEW: Temporal validity
    pub source_session_id: String,      // NEW: Provenance
    pub tags: Vec<String>,              // NEW: Better categorization
}

pub struct FactGraph {
    nodes: HashMap<String, FactNode>,
    edges: Vec<(String, String)>,  // Dependency graph
}

impl FactGraph {
    pub fn query_current(&self, category: &str) -> Option<&FactNode> {
        // Return only non-superseded facts from specified category
        self.nodes.values()
            .filter(|f| f.category == category && f.superseded_by.is_none())
            .max_by_key(|f| f.timestamp)
    }
}
GAP 3: Session Summarizer Operates in Isolation
Problem:

Current Implementation (mem-session/src/lib.rs):

Creates narrative summary but doesn't integrate it back into context
Only stores metadata, doesn't prune history
No validation that summary is accurate
Issue:

Rust
// Current: Stores in metadata only
meta.insert("narrative_summary".to_string(), serde_json::Value::String(summary.clone()));

// Missing: Doesn't actually replace context history with summary
// Comment says: "Production-ready iterative injection logic could go here"
Impact: Summarization layer doesn't actually compress context.

Recommendation:

Rust
pub struct SessionSummarizer {
    pub llm: Arc<dyn LlmClient>,
    pub interval: usize,
    pub compression_ratio: f32,  // NEW: Target 60% size reduction
    pub validation_mode: bool,    // NEW: Validate summary matches facts
}

#[async_trait]
impl MemoryLayer for SessionSummarizer {
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        if context.items.len() < self.interval {
            return Ok(());
        }

        // 1. Generate summary
        let summary = self.generate_summary(context).await?;
        
        // 2. Validate summary retains critical facts
        if self.validation_mode {
            self.validate_summary_fidelity(&summary, context).await?;
        }
        
        // 3. Replace older items with summary
        let cutoff_idx = context.items.len().saturating_sub(
            (context.items.len() as f32 * self.compression_ratio) as usize
        );
        
        let summary_item = MemoryItem {
            role: MemoryRole::System,
            content: summary,
            timestamp: chrono::Utc::now().timestamp() as u64,
            metadata: serde_json::json!({
                "type": "session_summary",
                "compressed_from": cutoff_idx,
                "original_item_count": context.items.len()
            }),
        };
        
        // Actually compress the context
        context.items = vec![vec![summary_item], context.items.drain(cutoff_idx..).collect()].concat();
        
        Ok(())
    }
    
    fn priority(&self) -> u32 {
        3
    }
}
GAP 4: Microcompactor TTL is Too Aggressive
Problem:

Current Implementation (mem-micro/src/lib.rs):

Rust
pub ttl_seconds: u64,  // default: 3600 (1 hour)

context.items.retain(|item| {
    match item.role {
        MemoryRole::System => true,  // Immortal
        _ => (now - item.timestamp) < self.ttl_seconds,
    }
});
Issues:

Linear TTL is too crude for complex reasoning
Loses important mid-conversation context
No concept of "relevance" vs "age"
System messages never expire (can accumulate garbage)
Impact: Agent loses recent context during long multi-turn conversations.

Recommendation:

Rust
pub struct AdaptiveMicroCompactor {
    pub base_ttl: u64,
    pub decay_function: TTLDecayStrategy,
    pub relevance_analyzer: Arc<dyn RelevanceAnalyzer>,  // NEW
}

pub enum TTLDecayStrategy {
    Linear { slope: f32 },
    Exponential { half_life: u64 },
    Gaussian { peak_age: u64, sigma: u64 },
    AdaptiveByType,  // Adjust TTL per message role/category
}

#[async_trait]
pub trait RelevanceAnalyzer: Send + Sync {
    /// Score how relevant a message is to current task (0.0 = irrelevant, 1.0 = critical)
    async fn score(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32>;
}

#[async_trait]
impl MemoryLayer for AdaptiveMicroCompactor {
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        let now = chrono::Utc::now().timestamp() as u64;
        
        for item in context.items.iter_mut() {
            let age = now.saturating_sub(item.timestamp);
            let relevance = self.relevance_analyzer.score(item, context).await?;
            
            // Adjust TTL based on relevance
            let effective_ttl = match &self.decay_function {
                TTLDecayStrategy::AdaptiveByType => {
                    // Keep high-relevance items longer
                    (self.base_ttl as f32 * (1.0 + relevance * 2.0)) as u64
                }
                TTLDecayStrategy::Exponential { half_life } => {
                    // Exponential decay still respects relevance boost
                    let decay = 2_f32.powf(-(age as f32) / (*half_life as f32));
                    (self.base_ttl as f32 * decay * (1.0 + relevance)) as u64
                }
                _ => self.base_ttl,
            };
            
            // Mark stale items for removal instead of removing directly
            if age > effective_ttl {
                item.metadata["stale"] = serde_json::json!(true);
            }
        }
        
        // Remove only truly stale items
        context.items.retain(|item| {
            !item.metadata.get("stale").map_or(false, |v| v.as_bool().unwrap_or(false))
        });
        
        Ok(())
    }
}
GAP 5: No Memory Reconstruction or Retrieval Mechanism
Problem:

Current Implementation:

Facts are stored in knowledge.json but there's no retrieval layer
Offloaded content hashes are stored but not indexed
No query interface to search/retrieve historical facts
Missing:

Semantic search over knowledge base
Retrieval by category or tags
RAG (Retrieval-Augmented Generation) integration
Memory injection at context formation time
Impact: Stored knowledge can't be used by the agent; defeats the purpose of extraction.

Recommendation:

Rust
pub struct MemoryRetriever<S: StorageBackend> {
    pub storage: S,
    pub embeddings: Arc<dyn EmbeddingProvider>,  // NEW
    pub index: VectorIndex,                      // NEW
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}

pub struct VectorIndex {
    facts: HashMap<String, (Fact, Vec<f32>)>,  // fact_id -> (fact, embedding)
}

impl MemoryRetriever {
    /// Retrieve relevant facts for current context (RAG integration)
    pub async fn retrieve_relevant_facts(
        &self,
        query: &str,
        top_k: usize,
        category_filter: Option<&str>,
    ) -> anyhow::Result<Vec<(Fact, f32)>> {
        let query_embedding = self.embeddings.embed(query).await?;
        
        let mut candidates: Vec<_> = self.index.facts.values()
            .filter(|(f, _)| {
                category_filter.is_none() || 
                category_filter == Some(&f.category)
            })
            .map(|(fact, embedding)| {
                let similarity = cosine_similarity(&query_embedding, embedding);
                (fact.clone(), similarity)
            })
            .collect();
        
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(candidates.into_iter().take(top_k).collect())
    }
    
    /// Reconstruct agent memory from facts when resuming
    pub async fn bootstrap_context_from_facts(&self) -> anyhow::Result<Context> {
        let facts = self.index.facts.values()
            .map(|(f, _)| f.clone())
            .collect::<Vec<_>>();
        
        let prompt = format!(
            "Reconstruct the agent's memory state from these facts:\n{}",
            facts.iter()
                .map(|f| format!("- [{}] {}", f.category, f.content))
                .collect::<Vec<_>>()
                .join("\n")
        );
        
        // Use LLM to synthesize coherent memory from facts
        Ok(Context {
            items: vec![/* reconstructed state */],
        })
    }
}
GAP 6: Full Compactor Uses Naive 80/20 Split
Problem:

Current Implementation (mem-compactor/src/lib.rs):

Rust
let keep_latest_count = context.items.len() / 5;  // Keep 20%
let split_idx = context.items.len().saturating_sub(keep_latest_count);
Issues:

Fixed split ignores information importance
Doesn't preserve decision points
May discard critical earlier context
No multi-level compression strategy
Example Problem:

Code
Turn 1-20: Setup & planning (crucial context)
Turn 21-95: Iterative refinement (intermediate)
Turn 96-100: Final details (keep per 80/20 rule)

Result: Setup lost, only final details + last 20%
Agent starts refining without understanding original goal
Recommendation:

Rust
pub struct IntelligentFullCompactor {
    pub llm: Arc<dyn LlmClient>,
    pub max_items: usize,
    pub importance_analyzer: Arc<dyn ImportanceAnalyzer>,  // NEW
    pub multi_level_compression: bool,                      // NEW
}

#[async_trait]
pub trait ImportanceAnalyzer: Send + Sync {
    async fn score_importance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32>;
}

#[async_trait]
impl MemoryLayer for IntelligentFullCompactor {
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        if context.items.len() < self.max_items {
            return Ok(());
        }

        // 1. Score each item by importance
        let mut scored_items: Vec<(usize, MemoryItem, f32)> = Vec::new();
        for (idx, item) in context.items.iter().enumerate() {
            let score = self.importance_analyzer.score_importance(item, context).await?;
            scored_items.push((idx, item.clone(), score));
        }

        // 2. Multi-level compression strategy
        if self.multi_level_compression {
            self.apply_multi_level_compression(&mut scored_items, context).await?;
        } else {
            // 3. Keep high-importance items, compress others
            let threshold = scored_items.iter()
                .map(|(_, _, s)| s)
                .cloned()
                .collect::<Vec<_>>();
            
            let top_20_pct = (threshold.len() / 5).max(1);
            scored_items.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            
            let high_importance: Vec<_> = scored_items.iter()
                .take(top_20_pct)
                .map(|(_, item, _)| item.clone())
                .collect();
            
            // Compress the rest
            let to_compress: Vec<_> = scored_items.iter()
                .skip(top_20_pct)
                .map(|(_, item, _)| item.clone())
                .collect();
            
            let compressed = self.compress_with_structure(&to_compress).await?;
            
            context.items = vec![compressed, high_importance].concat();
        }

        Ok(())
    }

    async fn compress_with_structure(&self, items: &[MemoryItem]) -> anyhow::Result<MemoryItem> {
        let prompt = r#"
        Apply 9-point structural summarization with special attention to:
        1. GOAL: Original user objective (MUST preserve)
        2. CONTEXT: Environmental facts
        3. TOOLS: Successfully used tools
        4. ERRORS: Technical failures that inform current approach
        5. PROGRESS: Percentage towards goal
        6. PENDING: Immediate next steps
        7. CONSTANTS: Immutable facts/constraints
        8. PREFERENCES: User formatting/style preferences
        9. NEXT: Very next action
        
        HISTORY: {history}
        "#.to_string();

        let history = items.iter()
            .map(|item| format!("{:?}: {}", item.role, item.content))
            .collect::<Vec<_>>()
            .join("\n");
        
        let summary = self.llm.completion(&prompt.replace("{history}", &history)).await?;
        
        Ok(MemoryItem {
            role: MemoryRole::System,
            content: format!("### COMPACTED SESSION STATE ###\n\n{}", summary),
            timestamp: chrono::Utc::now().timestamp() as u64,
            metadata: serde_json::json!({
                "compaction": true,
                "layer": 4,
                "compressed_item_count": items.len()
            }),
        })
    }
}
GAP 7: Dream Worker Has No Trigger or Scheduling
Problem:

Current Implementation (mem-dreamer/src/lib.rs):

run_dream_cycle() is defined but never automatically triggered
No background task scheduler
Requires manual invocation
Missing:

Idle detection mechanism
Scheduled background workers
Integration with agent lifecycle
Impact: Layer 6 (background consolidation) never runs automatically.

Recommendation:

Rust
use tokio::task::JoinHandle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub struct DreamScheduler<S: StorageBackend> {
    pub worker: Arc<DreamWorker<S>>,
    pub config: DreamConfig,
    pub last_activity: Arc<AtomicU64>,  // NEW: Track last user interaction
    pub handle: Option<JoinHandle<()>>,  // NEW: Background task handle
}

impl<S: StorageBackend> DreamScheduler<S> {
    pub fn new(worker: Arc<DreamWorker<S>>, config: DreamConfig) -> Self {
        Self {
            worker,
            config,
            last_activity: Arc::new(AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            )),
            handle: None,
        }
    }

    pub fn record_activity(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_activity.store(now, Ordering::Relaxed);
    }

    pub fn start_background_dreaming(&mut self) {
        let worker = Arc::clone(&self.worker);
        let last_activity = Arc::clone(&self.last_activity);
        let idle_threshold = self.config.idle_threshold_mins * 60;

        let handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;  // Check every minute

                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let last = last_activity.load(Ordering::Relaxed);
                let idle_time = now.saturating_sub(last);

                if idle_time > idle_threshold as u64 {
                    tracing::info!("Triggering dream cycle after {} seconds idle", idle_time);
                    if let Err(e) = worker.run_dream_cycle().await {
                        tracing::error!("Dream cycle failed: {:?}", e);
                    }
                    // Reset activity timer after successful dream
                    last_activity.store(now, Ordering::Relaxed);
                }
            }
        });

        self.handle = Some(handle);
    }

    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
        Ok(())
    }
}
GAP 8: No Conflict Resolution or Fact Validation
Problem:

Current Implementation:

Facts are stored as-is without validation
No mechanism to detect contradictions
No way to resolve conflicting facts
Example:

Code
Fact 1: "Database uses PostgreSQL" (confidence: 0.95)
Fact 2: "Database uses MySQL" (confidence: 0.88)

Both stored. No mechanism to resolve contradiction.
Recommendation:

Rust
pub struct ConflictResolver {
    pub llm: Arc<dyn LlmClient>,
    pub storage: Arc<dyn StorageBackend>,
}

impl ConflictResolver {
    pub async fn detect_and_resolve_conflicts(
        &self,
        kb: &mut KnowledgeBase,
    ) -> anyhow::Result<()> {
        let conflicts = self.detect_conflicts(kb)?;
        
        for conflict_group in conflicts {
            if conflict_group.len() <= 1 {
                continue;
            }

            tracing::warn!("Detected {} conflicting facts in category: {:?}",
                conflict_group.len(),
                conflict_group[0].category);

            let resolution = self.resolve_conflict(conflict_group.clone()).await?;
            
            match resolution {
                ConflictResolution::KeepHighestConfidence => {
                    let best = conflict_group.into_iter()
                        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                        .unwrap();
                    
                    kb.facts.retain(|f| f.category != best.category || f.confidence >= best.confidence);
                    kb.facts.push(best);
                }
                ConflictResolution::Merge(merged_fact) => {
                    kb.facts.retain(|f| f.category != merged_fact.category);
                    kb.facts.push(merged_fact);
                }
                ConflictResolution::Escalate => {
                    tracing::error!("Unresolvable conflict, escalating for manual review");
                }
            }
        }

        Ok(())
    }

    fn detect_conflicts(&self, kb: &KnowledgeBase) -> anyhow::Result<Vec<Vec<Fact>>> {
        let mut grouped: HashMap<String, Vec<Fact>> = HashMap::new();
        
        for fact in &kb.facts {
            grouped.entry(fact.category.clone())
                .or_insert_with(Vec::new)
                .push(fact.clone());
        }

        Ok(grouped.into_values()
            .filter(|group| group.len() > 1)
            .collect())
    }

    async fn resolve_conflict(&self, facts: Vec<Fact>) -> anyhow::Result<ConflictResolution> {
        let prompt = format!(
            "Resolve this conflict between facts:\n{}",
            facts.iter()
                .enumerate()
                .map(|(i, f)| format!(
                    "{}. [confidence: {}] {}",
                    i + 1, f.confidence, f.content
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let response = self.llm.completion(&prompt).await?;
        
        // Parse LLM response to determine resolution strategy
        // Implementation details...
        
        Ok(ConflictResolution::KeepHighestConfidence)
    }
}

pub enum ConflictResolution {
    KeepHighestConfidence,
    Merge(Fact),
    Escalate,
}
GAP 9: Insufficient Error Handling & Recovery
Problem:

Most error cases use .await? without recovery strategy
No graceful degradation
No circuit breaker pattern for LLM failures
Storage corruption not handled
Recommendation:

Rust
pub struct ResilientMemoryController<S: StorageBackend> {
    pub brain: Brain,
    pub storage: S,
    pub error_budget: Arc<Mutex<ErrorBudget>>,
    pub circuit_breaker: Arc<CircuitBreaker>,
}

pub struct ErrorBudget {
    pub max_failures: usize,
    pub failures: usize,
    pub window: Duration,
    pub last_reset: Instant,
}

pub struct CircuitBreaker {
    pub state: CircuitState,
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub timeout: Duration,
}

pub enum CircuitState {
    Closed,           // Normal operation
    Open,             // Reject requests
    HalfOpen,         // Testing recovery
}

#[async_trait]
pub trait Recoverable<T> {
    async fn with_recovery(&self) -> anyhow::Result<T>;
}

impl ResilientMemoryController {
    pub async fn optimize_resilient(&self, context: &mut Context) -> anyhow::Result<()> {
        let mut budget = self.error_budget.lock().await;
        
        if budget.failures > budget.max_failures {
            tracing::warn!("Error budget exceeded, operating in degraded mode");
            return Ok(());
        }

        match self.brain.optimize(context).await {
            Ok(_) => {
                budget.failures = 0;
                Ok(())
            }
            Err(e) => {
                budget.failures += 1;
                tracing::error!("Memory optimization failed (attempt {}): {:?}", 
                    budget.failures, e);
                
                if budget.failures >= budget.max_failures {
                    // Escalate: Create backup snapshot
                    self.create_emergency_snapshot(context).await?;
                }
                
                Err(e)
            }
        }
    }

    async fn create_emergency_snapshot(&self, context: &Context) -> anyhow::Result<()> {
        let snapshot_id = format!("emergency_snapshot_{}", chrono::Utc::now().timestamp());
        let data = serde_json::to_vec(context)?;
        self.storage.store(&snapshot_id, &data).await?;
        tracing::info!("Created emergency snapshot: {}", snapshot_id);
        Ok(())
    }
}
GAP 10: Missing Observability & Metrics
Problem:

No instrumentation for monitoring memory health
Can't track compression ratios, cache hit rates, etc.
No visibility into layer performance
Recommendation:

Rust
pub struct MemoryMetrics {
    pub context_size_bytes: prometheus::Gauge,
    pub item_count: prometheus::Gauge,
    pub cache_hit_rate: prometheus::Gauge,
    pub compression_ratio: prometheus::Gauge,
    pub layer_latency: HashMap<String, prometheus::Histogram>,
    pub fact_count: prometheus::Gauge,
    pub knowledge_base_size: prometheus::Gauge,
}

impl Brain {
    pub async fn optimize_with_metrics(
        &self,
        context: &mut Context,
        metrics: &MemoryMetrics,
    ) -> anyhow::Result<()> {
        let initial_size = serde_json::to_vec(context)?.len();
        let initial_items = context.items.len();

        for layer in &self.layers {
            let start = Instant::now();
            layer.process(context).await?;
            let duration = start.elapsed();

            if let Some(histogram) = metrics.layer_latency.get(layer.name()) {
                histogram.observe(duration.as_secs_f64());
            }
        }

        let final_size = serde_json::to_vec(context)?.len();
        let final_items = context.items.len();

        metrics.context_size_bytes.set(final_size as f64);
        metrics.item_count.set(final_items as f64);
        metrics.compression_ratio.set(
            (final_size as f64 / initial_size as f64) * 100.0
        );

        Ok(())
    }
}