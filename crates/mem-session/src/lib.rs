use mem_core::{MemoryLayer, Context, LlmClient};
use async_trait::async_trait;
use std::sync::Arc;

pub struct SessionSummarizer {
    pub llm: Arc<dyn LlmClient>,
    /// How often to run the summarizer (number of new items)
    pub interval: usize,
}

impl SessionSummarizer {
    pub fn new(llm: Arc<dyn LlmClient>, interval: usize) -> Self {
        Self { llm, interval }
    }
}

#[async_trait]
impl MemoryLayer for SessionSummarizer {
    fn name(&self) -> &str {
        "SessionSummarizer"
    }

    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        // Only run if we hit the interval threshold
        if context.items.len() < self.interval {
            return Ok(());
        }

        let mut history = String::new();
        for item in &context.items {
            history.push_str(&format!("{:?}: {}\n", item.role, item.content));
        }

        let prompt = format!(
            "Analyze the conversation history below and provide a concise, high-level markdown summary of the progress and current state.\n\n\
            HISTORY:\n{}",
            history
        );

        let summary = self.llm.completion(&prompt).await?;

        // Update the metadata of the last item to include this "Narrative Thread"
        if let Some(last_item) = context.items.last_mut() {
            let mut meta = last_item.metadata.as_object_mut().cloned().unwrap_or_default();
            meta.insert("narrative_summary".to_string(), serde_json::Value::String(summary));
            last_item.metadata = serde_json::Value::Object(meta);
        }

        Ok(())
    }

    fn priority(&self) -> u32 {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mem_core::{MemoryItem, MemoryRole};

    struct MockLlm;
    #[async_trait]
    impl LlmClient for MockLlm {
        async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
            Ok("Summary: User and Assistant discussed memory.".to_string())
        }
    }

    #[tokio::test]
    async fn test_session_summarizer_trigger() {
        let llm = Arc::new(MockLlm);
        let summarizer = SessionSummarizer::new(llm, 2);
        
        let mut context = Context {
            items: vec![
                MemoryItem {
                    role: MemoryRole::User,
                    content: "Hello".to_string(),
                    timestamp: 0,
                    metadata: serde_json::json!({}),
                },
                MemoryItem {
                    role: MemoryRole::Assistant,
                    content: "Hi there".to_string(),
                    timestamp: 1,
                    metadata: serde_json::json!({}),
                },
            ],
        };

        summarizer.process(&mut context).await.unwrap();

        // Verify narrative summary was added to the last item's metadata
        let last_meta = &context.items[1].metadata;
        assert!(last_meta["narrative_summary"].as_str().is_some());
        assert_eq!(last_meta["narrative_summary"], "Summary: User and Assistant discussed memory.");
    }
}
