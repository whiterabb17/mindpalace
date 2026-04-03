use crate::{RelevanceAnalyzer, ImportanceAnalyzer, MemoryItem, Context, LlmClient};
use async_trait::async_trait;
use std::sync::Arc;

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
