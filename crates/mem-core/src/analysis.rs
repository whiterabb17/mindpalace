use crate::{RelevanceAnalyzer, ImportanceAnalyzer, MemoryItem, Context, LlmClient};
use async_trait::async_trait;
use std::sync::Arc;

/// A relevance analyzer that uses an LLM to score the relationship between a message and its context.
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
        if context.items.is_empty() { return Ok(1.0); }
        
        // Take last 3 items for context
        let last_items: Vec<_> = context.items.iter().rev().take(3).collect();
        let mut recent_context = String::new();
        for i in last_items.into_iter().rev() {
            recent_context.push_str(&format!("{:?}: {}\n", i.role, i.content));
        }

        let prompt = format!(
            "Score message relevance (0.0 to 1.0) to current context.\n\nCONTEXT:\n{}\n\nMESSAGE:\n{}\n\nReturn ONLY the number.",
            recent_context, item.content
        );

        let response = self.llm.completion(&prompt).await?;
        Ok(response.trim().parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0))
    }
}

/// An importance analyzer that uses an LLM to determine the value of a message for long-term retention.
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

        let response = self.llm.completion(&prompt).await?;
        Ok(response.trim().parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0))
    }
}

/// A fast, rule-based importance analyzer for local decision making without LLM latency.
pub struct HeuristicImportanceAnalyzer;

#[async_trait]
impl ImportanceAnalyzer for HeuristicImportanceAnalyzer {
    async fn score_importance(&self, item: &MemoryItem, _context: &Context) -> anyhow::Result<f32> {
        let mut score: f32 = match item.role {
            crate::MemoryRole::System => 1.0,
            crate::MemoryRole::Assistant => 0.7,
            crate::MemoryRole::User => 0.6,
            crate::MemoryRole::Tool => 0.5,
        };

        let content_lower = item.content.to_lowercase();
        
        // Semantic triggers (High Importance)
        let high_keywords = ["goal", "decision", "finally", "fixed", "constraint", "v14", "sql", "architecture", "fact"];
        if high_keywords.iter().any(|k| content_lower.contains(k)) {
            score += 0.2;
        }

        // Action items (Medium Importance)
        let action_keywords = ["todo", "next", "implement", "refactor", "check"];
        if action_keywords.iter().any(|k| content_lower.contains(k)) {
            score += 0.1;
        }

        // Noise reduction (Low Importance)
        let noise_keywords = ["hello", "thanks", "ok", "cool", "wait"];
        if noise_keywords.iter().any(|k| content_lower.contains(k)) {
            score -= 0.1;
        }

        Ok(score.clamp(0.0, 1.0))
    }
}

/// A deterministic relevance analyzer based on keyword overlap.
pub struct KeywordRelevanceAnalyzer;

#[async_trait]
impl RelevanceAnalyzer for KeywordRelevanceAnalyzer {
    async fn score_relevance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32> {
        let last_msg = context.items.last().map(|i| i.content.to_lowercase()).unwrap_or_default();
        if last_msg.is_empty() { return Ok(1.0); }
        
        let content_lower = item.content.to_lowercase();
        let query_tokens: std::collections::HashSet<_> = last_msg.split_whitespace().collect();
        let msg_tokens: std::collections::HashSet<_> = content_lower.split_whitespace().collect();
        
        let intersection = query_tokens.intersection(&msg_tokens).count();
        if query_tokens.is_empty() { return Ok(0.5); }
        
        Ok((intersection as f32 / query_tokens.len() as f32).clamp(0.0, 1.0))
    }
}

/// A vector-based relevance analyzer using semantic embeddings.
pub struct VectorRelevanceAnalyzer {
    pub embeddings: Arc<dyn crate::EmbeddingProvider>,
}

#[async_trait]
impl RelevanceAnalyzer for VectorRelevanceAnalyzer {
    async fn score_relevance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32> {
        let last_msg = context.items.last().map(|i| i.content.as_str()).unwrap_or("");
        if last_msg.is_empty() { return Ok(1.0); }
        
        let query_vec = self.embeddings.embed(last_msg).await?;
        let msg_vec = self.embeddings.embed(&item.content).await?;
        
        Ok(crate::utils::cosine_similarity(&query_vec, &msg_vec).clamp(0.0, 1.0))
    }
}
