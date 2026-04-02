use mem_core::{MemoryLayer, Context, LlmClient, MemoryItem, MemoryRole};
use async_trait::async_trait;
use std::sync::Arc;

pub struct FullCompactor {
    pub llm: Arc<dyn LlmClient>,
    /// Maximum items allowed before hard compaction triggers
    pub max_items: usize,
}

impl FullCompactor {
    pub fn new(llm: Arc<dyn LlmClient>, max_items: usize) -> Self {
        Self { llm, max_items }
    }
}

pub const COMPACTION_PROMPT: &str = r#"
Provide a 9-point structural summary of the current agent session. 
Be precise and technical.

1. GOAL: Current objective.
2. CONTEXT: Environmental constraints.
3. TOOLS: Successfully used tools.
4. ERRORS: Technical failures encountered.
5. PROGRESS: Percentage towards current goal.
6. PENDING: Immediate next steps.
7. CONSTANTS: Facts that must not change.
8. PREFERENCES: Explicit user formatting instructions.
9. NEXT: The very next requested action.

HISTORY:
{history}
"#;

#[async_trait]
impl MemoryLayer for FullCompactor {
    fn name(&self) -> &str {
        "FullCompactor"
    }

    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        if context.items.len() < self.max_items {
            return Ok(());
        }

        let mut history = String::new();
        for item in &context.items {
            history.push_str(&format!("{:?}: {}\n", item.role, item.content));
        }

        let prompt = COMPACTION_PROMPT.replace("{history}", &history);
        let summary_text = self.llm.completion(&prompt).await?;

        // 80/20 Pruning: Replace first 80% with the summary, keep latest 20%
        let keep_latest_count = context.items.len() / 5; 
        
        let summary_item = MemoryItem {
            role: MemoryRole::System,
            content: format!("### COMPACTED SESSION STATE ###\n\n{}", summary_text),
            timestamp: chrono::Utc::now().timestamp() as u64,
            metadata: serde_json::json!({"compaction": true, "layer": 4}),
        };

        let mut new_items = vec![summary_item];
        let split_idx = context.items.len().saturating_sub(keep_latest_count);
        
        // Drain everything up to the split point, keep the rest
        let latest_items: Vec<_> = context.items.drain(split_idx..).collect();
        new_items.extend(latest_items);
        
        context.items = new_items;

        Ok(())
    }

    fn priority(&self) -> u32 {
        4
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
            Ok("9-Point Summary Data".to_string())
        }
    }

    #[tokio::test]
    async fn test_full_compaction_logical_split() {
        let llm = Arc::new(MockLlm);
        let compactor = FullCompactor::new(llm, 5); // Trigger at 5 items
        
        let mut context = Context {
            items: (0..5).map(|i| MemoryItem {
                role: MemoryRole::User,
                content: format!("Message {}", i),
                timestamp: i as u64,
                metadata: serde_json::json!({}),
            }).collect(),
        };

        compactor.process(&mut context).await.unwrap();

        // 80/20 Rule:
        // Original: 5 items. 
        // split_idx = 5 - (5/5) = 4. 
        // We keep 1 latest item (Message 4) and replace others with 1 summary.
        // Total should be 2 items.
        assert_eq!(context.items.len(), 2);
        assert!(context.items[0].content.contains("COMPACTED SESSION STATE"));
        assert_eq!(context.items[1].content, "Message 4");
    }
}
