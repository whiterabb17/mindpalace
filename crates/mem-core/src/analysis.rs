use crate::{Context, ImportanceAnalyzer, LlmClient, MemoryItem, MemoryRole, RelevanceAnalyzer};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::metrics::AnalysisMetrics;

// --- Float Parsing Helper ---
pub async fn parse_llm_score(response: &str, context: &str) -> anyhow::Result<f32> {
    let trimmed = response.trim();

    // Try to extract float from various formats
    let score = if trimmed.starts_with('[') && trimmed.ends_with(']') {
        // Handle JSON array responses
        trimmed[1..trimmed.len() - 1].parse::<f32>()?
    } else if let Ok(val) = trimmed.parse::<f32>() {
        val
    } else {
        // Try regex extraction for "Score: 0.75" format
        regex::Regex::new(r"(\d+\.?\d*)")?
            .captures(trimmed)
            .and_then(|c| c.get(1).map(|m| m.as_str()))
            .ok_or_else(|| anyhow::anyhow!("No valid score found in: {}", trimmed))?
            .parse::<f32>()?
    };

    if !(0.0..=1.0).contains(&score) {
        anyhow::bail!("Score {} outside valid range [0.0, 1.0]", score);
    }

    tracing::debug!(
        target: "mem-core::analysis",
        context = %context,
        raw_response = %trimmed,
        parsed_score = score,
        "LLM score parsed successfully"
    );

    Ok(score)
}

// --- Timeouts & Context ---
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisTimeoutConfig {
    pub keyword_analyzer_timeout: Duration,
    pub heuristic_analyzer_timeout: Duration,
    pub vector_analyzer_timeout: Duration,
    pub llm_analyzer_timeout: Duration,
    pub fallback_chain_timeout: Duration,
    pub circuit_breaker_reset: Duration,
    pub cache_ttl: Duration,
}

impl AnalysisTimeoutConfig {
    pub fn new() -> Self {
        Self {
            keyword_analyzer_timeout: Duration::from_millis(500),
            heuristic_analyzer_timeout: Duration::from_millis(200),
            vector_analyzer_timeout: Duration::from_secs(8),
            llm_analyzer_timeout: Duration::from_secs(12),
            fallback_chain_timeout: Duration::from_secs(20),
            circuit_breaker_reset: Duration::from_secs(45),
            cache_ttl: Duration::from_secs(3600),
        }
    }

    pub fn from_env() -> Self {
        let mut config = Self::new();
        if let Ok(val) = std::env::var("MINDPALACE_KEYWORD_TIMEOUT_MS") {
            if let Ok(ms) = val.parse::<u64>() {
                config.keyword_analyzer_timeout = Duration::from_millis(ms);
            }
        }
        if let Ok(val) = std::env::var("MINDPALACE_HEURISTIC_TIMEOUT_MS") {
            if let Ok(ms) = val.parse::<u64>() {
                config.heuristic_analyzer_timeout = Duration::from_millis(ms);
            }
        }
        if let Ok(val) = std::env::var("MINDPALACE_VECTOR_TIMEOUT_SECS") {
            if let Ok(secs) = val.parse::<u64>() {
                config.vector_analyzer_timeout = Duration::from_secs(secs);
            }
        }
        if let Ok(val) = std::env::var("MINDPALACE_LLM_TIMEOUT_SECS") {
            if let Ok(secs) = val.parse::<u64>() {
                config.llm_analyzer_timeout = Duration::from_secs(secs);
            }
        }
        if let Ok(val) = std::env::var("MINDPALACE_FALLBACK_CHAIN_TIMEOUT_SECS") {
            if let Ok(secs) = val.parse::<u64>() {
                config.fallback_chain_timeout = Duration::from_secs(secs);
            }
        }
        if let Ok(val) = std::env::var("MINDPALACE_CIRCUIT_BREAKER_RESET_SECS") {
            if let Ok(secs) = val.parse::<u64>() {
                config.circuit_breaker_reset = Duration::from_secs(secs);
            }
        }
        config
    }

    pub fn for_context(&self, ctx: &AnalysisContext) -> Duration {
        match ctx {
            AnalysisContext::Interactive => self.llm_analyzer_timeout / 2,
            AnalysisContext::Background => self.llm_analyzer_timeout * 2,
            AnalysisContext::Degraded => self.heuristic_analyzer_timeout,
        }
    }
}

impl Default for AnalysisTimeoutConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalysisContext {
    Interactive,
    Background,
    Degraded,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalyzerStrategy {
    Full,
    VectorOnly,
    HeuristicOnly,
}

pub struct AnalyzerHealthMonitor {
    llm_timeouts: Arc<AtomicU32>,
    vector_timeouts: Arc<AtomicU32>,
    timeout_threshold: u32,
    pub config: AnalysisTimeoutConfig,
}

impl AnalyzerHealthMonitor {
    pub fn new(config: AnalysisTimeoutConfig) -> Self {
        Self {
            llm_timeouts: Arc::new(AtomicU32::new(0)),
            vector_timeouts: Arc::new(AtomicU32::new(0)),
            timeout_threshold: 3,
            config,
        }
    }

    pub fn get_active_analyzer(&self) -> AnalyzerStrategy {
        let llm_failures = self.llm_timeouts.load(Ordering::Relaxed);
        let vector_failures = self.vector_timeouts.load(Ordering::Relaxed);

        if llm_failures >= self.timeout_threshold {
            tracing::warn!(llm_failures, "LLM analyzer degraded, using vector fallback");
            AnalyzerStrategy::VectorOnly
        } else if vector_failures >= self.timeout_threshold {
            tracing::warn!(
                vector_failures,
                "Vector analyzer degraded, using heuristic fallback"
            );
            AnalyzerStrategy::HeuristicOnly
        } else {
            AnalyzerStrategy::Full
        }
    }

    pub fn record_llm_timeout(&self) {
        self.llm_timeouts.fetch_add(1, Ordering::SeqCst);
    }
    pub fn record_llm_success(&self) {
        self.llm_timeouts.store(0, Ordering::SeqCst);
    }
    pub fn record_vector_timeout(&self) {
        self.vector_timeouts.fetch_add(1, Ordering::SeqCst);
    }
    pub fn record_vector_success(&self) {
        self.vector_timeouts.store(0, Ordering::SeqCst);
    }
}

// --- Analyzers ---

const MAX_CONTEXT_DISPLAY: usize = 2048;
const MAX_ITEM_SIZE: usize = 8192;

async fn build_relevance_context(
    context: &Context,
    max_items: usize,
    max_chars: usize,
) -> anyhow::Result<String> {
    let mut recent_context = String::with_capacity(max_chars);
    let mut total_chars = 0;

    for item in context.items.iter().rev().take(max_items) {
        if item.content.len() > MAX_ITEM_SIZE {
            tracing::warn!(
                item_size = item.content.len(),
                max_allowed = MAX_ITEM_SIZE,
                "Item exceeds maximum size, truncating"
            );
        }

        let role_str = match item.role {
            MemoryRole::System => "SYSTEM",
            MemoryRole::Assistant => "ASSISTANT",
            MemoryRole::User => "USER",
            MemoryRole::Tool => "TOOL",
        };

        let content_preview = &item.content[..item.content.len().min(256)];
        let line = format!("{}: {}\n", role_str, content_preview);

        if total_chars + line.len() > max_chars {
            tracing::debug!("Context window full, stopping");
            recent_context.push_str("...(truncated)");
            break;
        }

        recent_context.push_str(&line);
        total_chars += line.len();
    }

    Ok(recent_context)
}

pub struct LlmRelevanceAnalyzer {
    pub llm: Arc<dyn LlmClient>,
}
impl LlmRelevanceAnalyzer {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl RelevanceAnalyzer for LlmRelevanceAnalyzer {
    async fn score_relevance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32> {
        if context.items.is_empty() {
            return Ok(1.0);
        }
        let recent_context = build_relevance_context(context, 3, MAX_CONTEXT_DISPLAY).await?;
        let prompt = format!(
            "Score message relevance (0.0 to 1.0) to current context.\n\nCONTEXT:\n{}\n\nMESSAGE:\n{}\n\nReturn ONLY the number.",
            recent_context, item.content
        );

        match self.llm.completion(&prompt).await {
            Ok(response) => Ok(parse_llm_score(&response, "LlmRelevanceAnalyzer").await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Failed to parse LLM relevance score, defaulting to 0.5");
                0.5
            })),
            Err(e) => {
                tracing::warn!(error = %e, "LLM provider failed for relevance scoring, defaulting to 0.5");
                Ok(0.5)
            }
        }
    }
}

pub struct LlmImportanceAnalyzer {
    pub llm: Arc<dyn LlmClient>,
}
impl LlmImportanceAnalyzer {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl ImportanceAnalyzer for LlmImportanceAnalyzer {
    async fn score_importance(&self, item: &MemoryItem, _context: &Context) -> anyhow::Result<f32> {
        let prompt = format!(
            "Score message importance for long-term retention (0.0 to 1.0). High for goals/decisions/facts.\n\nMESSAGE: {}\n\nReturn ONLY the number.",
            item.content
        );

        match self.llm.completion(&prompt).await {
            Ok(response) => Ok(parse_llm_score(&response, "LlmImportanceAnalyzer").await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Failed to parse LLM importance score, defaulting to 0.5");
                0.5
            })),
            Err(e) => {
                tracing::warn!(error = %e, "LLM provider failed for importance scoring, defaulting to 0.5");
                Ok(0.5)
            }
        }
    }
}

pub struct KeywordMatcher {
    high_importance: Regex,
    medium_importance: Regex,
    low_importance: Regex,
}

impl Default for KeywordMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl KeywordMatcher {
    pub fn new() -> Self {
        Self {
            high_importance: Regex::new(
                r"(?i)\b(goal|decision|finally|fixed|constraint|sql|architecture|fact)\b",
            )
            .unwrap(),
            medium_importance: Regex::new(
                r"(?i)\b(todo|next|implement|refactor|check|bug|issue)\b",
            )
            .unwrap(),
            low_importance: Regex::new(r"(?i)\b(hello|thanks|ok|cool|wait|hmm|sure)\b").unwrap(),
        }
    }

    pub fn score_content(&self, content: &str) -> f32 {
        let mut score = 0.0f32;
        let high_matches = self.high_importance.find_iter(content).count();
        let medium_matches = self.medium_importance.find_iter(content).count();
        let low_matches = self.low_importance.find_iter(content).count();

        score += (high_matches as f32) * 0.15;
        score += (medium_matches as f32) * 0.08;
        score -= (low_matches as f32) * 0.05;
        score.clamp(-1.0, 1.0)
    }
}

static KEYWORD_PATTERNS: Lazy<KeywordMatcher> = Lazy::new(KeywordMatcher::new);

pub struct HeuristicImportanceAnalyzer;

#[async_trait]
impl ImportanceAnalyzer for HeuristicImportanceAnalyzer {
    async fn score_importance(&self, item: &MemoryItem, _context: &Context) -> anyhow::Result<f32> {
        let mut score: f32 = match item.role {
            MemoryRole::System => 1.0,
            MemoryRole::Assistant => 0.7,
            MemoryRole::User => 0.6,
            MemoryRole::Tool => 0.5,
        };
        score += KEYWORD_PATTERNS.score_content(&item.content);
        Ok(score.clamp(0.0, 1.0))
    }
}

const MAX_TOKENS_PER_MESSAGE: usize = 500;

pub struct KeywordRelevanceAnalyzer;

impl KeywordRelevanceAnalyzer {
    fn tokenize_with_limits(&self, text: &str, max_tokens: usize) -> HashSet<String> {
        text.split_whitespace()
            .take(max_tokens)
            .map(|t| t.to_lowercase())
            .filter(|token| token.len() > 2)
            .collect()
    }
}

#[async_trait]
impl RelevanceAnalyzer for KeywordRelevanceAnalyzer {
    async fn score_relevance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32> {
        if context.items.is_empty() {
            return Ok(0.5);
        }
        let recent_messages: Vec<_> = context.items.iter().rev().take(3).collect();
        let mut max_relevance = 0.0f32;

        for msg in recent_messages {
            let query_tokens = self.tokenize_with_limits(&msg.content, MAX_TOKENS_PER_MESSAGE);
            let item_tokens = self.tokenize_with_limits(&item.content, MAX_TOKENS_PER_MESSAGE);

            if query_tokens.is_empty() {
                continue;
            }
            let intersection = query_tokens.intersection(&item_tokens).count();
            let relevance = (intersection as f32) / (query_tokens.len() as f32);
            max_relevance = max_relevance.max(relevance);
        }
        Ok(max_relevance.clamp(0.0, 1.0))
    }
}

pub struct CachedVectorRelevanceAnalyzer {
    pub embeddings: Arc<dyn crate::EmbeddingProvider>,
    cache: moka::future::Cache<String, Vec<f32>>,
}

impl CachedVectorRelevanceAnalyzer {
    pub fn new(embeddings: Arc<dyn crate::EmbeddingProvider>) -> Self {
        Self {
            embeddings,
            cache: moka::future::Cache::builder()
                .max_capacity(1000)
                .time_to_idle(Duration::from_secs(3600))
                .build(),
        }
    }

    async fn get_embedding_cached(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let key = text.to_string();
        if let Some(vec) = self.cache.get(&key).await {
            return Ok(vec);
        }
        let vec =
            tokio::time::timeout(Duration::from_secs(10), self.embeddings.embed(text)).await??;
        self.cache.insert(key, vec.clone()).await;
        Ok(vec)
    }
}

#[async_trait]
impl RelevanceAnalyzer for CachedVectorRelevanceAnalyzer {
    async fn score_relevance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32> {
        let last_msg = context
            .items
            .last()
            .map(|i| i.content.as_str())
            .unwrap_or("");
        if last_msg.is_empty() {
            return Ok(0.5);
        }

        let query_vec = self.get_embedding_cached(last_msg).await?;
        let msg_vec = self.get_embedding_cached(&item.content).await?;
        Ok(crate::utils::cosine_similarity(&query_vec, &msg_vec).clamp(0.0, 1.0))
    }
}

pub struct FallbackRelevanceAnalyzer {
    primary: Arc<dyn RelevanceAnalyzer>,
    fallback: Arc<dyn RelevanceAnalyzer>,
}

impl FallbackRelevanceAnalyzer {
    pub fn new(primary: Arc<dyn RelevanceAnalyzer>, fallback: Arc<dyn RelevanceAnalyzer>) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl RelevanceAnalyzer for FallbackRelevanceAnalyzer {
    async fn score_relevance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32> {
        match tokio::time::timeout(
            Duration::from_secs(5),
            self.primary.score_relevance(item, context),
        )
        .await
        {
            Ok(Ok(score)) => {
                tracing::debug!("Primary analyzer returned score: {}", score);
                Ok(score)
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "Primary analyzer failed, falling back to secondary");
                self.fallback.score_relevance(item, context).await
            }
            Err(_) => {
                tracing::warn!("Primary analyzer timed out, using fallback");
                self.fallback.score_relevance(item, context).await
            }
        }
    }
}

pub struct InstrumentedAnalyzer {
    pub primary: Arc<dyn RelevanceAnalyzer>,
    pub fallback: Arc<dyn RelevanceAnalyzer>,
    pub metrics: Arc<AnalysisMetrics>,
    pub config: AnalysisTimeoutConfig,
    pub health_monitor: Arc<AnalyzerHealthMonitor>,
}

#[async_trait]
impl RelevanceAnalyzer for InstrumentedAnalyzer {
    async fn score_relevance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32> {
        let strategy = self.health_monitor.get_active_analyzer();

        match strategy {
            AnalyzerStrategy::Full => {
                let start = std::time::Instant::now();
                match tokio::time::timeout(
                    self.config.llm_analyzer_timeout,
                    self.primary.score_relevance(item, context),
                )
                .await
                {
                    Ok(Ok(score)) => {
                        let duration = start.elapsed();
                        self.metrics
                            .llm_analyzer_latency_secs
                            .observe(duration.as_secs_f64());
                        self.metrics.relevance_scores.observe(score as f64);
                        self.health_monitor.record_llm_success();

                        tracing::debug!(
                            analyzer = "llm",
                            score = score,
                            latency_ms = duration.as_millis(),
                            "Relevance scored"
                        );

                        Ok(score)
                    }
                    Ok(Err(e)) => {
                        self.metrics.llm_failures_total.inc();
                        self.health_monitor.record_llm_timeout();

                        tracing::warn!(
                            analyzer = "llm",
                            error = %e,
                            "LLM analyzer failed, falling back"
                        );

                        self.fallback.score_relevance(item, context).await
                    }
                    Err(_) => {
                        self.metrics.llm_timeout_count.inc();
                        self.health_monitor.record_llm_timeout();

                        tracing::warn!(
                            analyzer = "llm",
                            timeout_secs = self.config.llm_analyzer_timeout.as_secs(),
                            "LLM analyzer timed out"
                        );

                        self.fallback.score_relevance(item, context).await
                    }
                }
            }

            AnalyzerStrategy::VectorOnly | AnalyzerStrategy::HeuristicOnly => {
                let start = std::time::Instant::now();
                let score = self.fallback.score_relevance(item, context).await?;
                let duration = start.elapsed();

                self.metrics.relevance_scores.observe(score as f64);

                tracing::debug!(
                    analyzer = "fallback",
                    strategy = ?strategy,
                    score = score,
                    latency_ms = duration.as_millis(),
                    "Relevance scored (degraded)"
                );

                Ok(score)
            }
        }
    }
}
