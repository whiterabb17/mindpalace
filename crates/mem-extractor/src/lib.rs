use mem_core::{MemoryLayer, Context, LlmClient, Fact, KnowledgeBase, StorageBackend};
use async_trait::async_trait;
use std::sync::Arc;
use serde_json;

pub struct FactExtractor<S: StorageBackend> {
    pub llm: Arc<dyn LlmClient>,
    pub storage: S,
    pub knowledge_path: String,
}

impl<S: StorageBackend> FactExtractor<S> {
    pub fn new(llm: Arc<dyn LlmClient>, storage: S, knowledge_path: String) -> Self {
        Self {
            llm,
            storage,
            knowledge_path,
        }
    }

    /// Extract facts using the LLM from the current context history.
    pub async fn extract_facts(&self, context: &Context) -> anyhow::Result<Vec<Fact>> {
        let mut history = String::new();
        for item in &context.items {
            history.push_str(&format!("{:?}: {}\n", item.role, item.content));
        }

        let prompt = format!(
            "Analyze the conversation history and extract durable, high-confidence facts about the user's intent, the project settings, and hardcoded knowledge. \
            Return the result as a simple JSON array of objects with the keys 'category', 'content', and 'confidence' (0.0 to 1.0). \
            Do not include any other text in your response.\n\n\
            HISTORY:\n{}",
            history
        );

        let response = self.llm.completion(&prompt).await?;
        
        // Basic JSON cleaning if LLM surrounds with code blocks
        let cleaned_response = response.trim().trim_start_matches("```json").trim_end_matches("```").trim();
        
        // Map anonymous JSON to Fact structs, adding current timestamp
        let raw_facts: Vec<serde_json::Value> = serde_json::from_str(cleaned_response)?;
        let facts: Vec<Fact> = raw_facts.into_iter().filter_map(|v| {
            if let (Some(cat), Some(cont), Some(conf)) = (v["category"].as_str(), v["content"].as_str(), v["confidence"].as_f64()) {
                Some(Fact {
                    category: cat.to_string(),
                    content: cont.to_string(),
                    confidence: conf as f32,
                    timestamp: chrono::Utc::now().timestamp() as u64,
                })
            } else {
                None
            }
        }).collect();

        Ok(facts)
    }

    /// Commit new facts to the persistent knowledge base, avoiding exact duplicates.
    pub async fn commit_knowledge(&self, new_facts: Vec<Fact>) -> anyhow::Result<()> {
        let mut kb = if self.storage.exists(&self.knowledge_path).await {
            let data = self.storage.retrieve(&self.knowledge_path).await?;
            serde_json::from_slice(&data).unwrap_or_default()
        } else {
            KnowledgeBase::default()
        };

        for fact in new_facts {
            // Simple string-based deduplication
            if !kb.facts.iter().any(|f| f.content.to_lowercase() == fact.content.to_lowercase()) {
                kb.facts.push(fact);
            }
        }

        let data = serde_json::to_vec_pretty(&kb)?;
        self.storage.store(&self.knowledge_path, &data).await?;
        Ok(())
    }
}

#[async_trait]
impl<S: StorageBackend> MemoryLayer for FactExtractor<S> {
    fn name(&self) -> &str {
        "FactExtractor"
    }

    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        // Milestone-based Trigger: Follow the Anthropic design of intentional extraction.
        // The orchestrator marks a milestone (e.g., goal reached) to trigger this layer.
        let is_milestone = context.items.last().map_or(false, |item| {
            item.metadata["milestone"].as_bool().unwrap_or(false)
        });

        if is_milestone {
            let facts = self.extract_facts(context).await?;
            self.commit_knowledge(facts).await?;
        }

        Ok(())
    }

    fn priority(&self) -> u32 {
        5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mem_core::{FileStorage, MemoryItem, MemoryRole};
    use tempfile::tempdir;

    struct MockLlm;
    #[async_trait]
    impl LlmClient for MockLlm {
        async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
            Ok(r#"[{"category": "user_preference", "content": "Prefers Rust", "confidence": 0.9}]"#.to_string())
        }
    }

    #[tokio::test]
    async fn test_fact_extraction_and_persistence() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path().to_path_buf());
        let llm = Arc::new(MockLlm);
        let extractor = FactExtractor::new(llm, storage, "knowledge.json".to_string());

        let context = Context {
            items: vec![
                MemoryItem {
                    role: MemoryRole::User,
                    content: "I really like using Rust for systems programming.".to_string(),
                    timestamp: 0,
                    metadata: serde_json::json!({"milestone": true}),
                }
            ],
        };

        // 1. Manually trigger extraction
        let facts = extractor.extract_facts(&context).await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "Prefers Rust");

        // 2. Commit to storage
        extractor.commit_knowledge(facts).await.unwrap();
        
        // 3. Verify file exists and has content
        assert!(dir.path().join("knowledge.json").exists());
        
        // 4. Verify deduplication
        let new_facts = vec![Fact {
            category: "user_preference".to_string(),
            content: "Prefers Rust".to_string(),
            confidence: 0.9,
            timestamp: 1,
        }];
        extractor.commit_knowledge(new_facts).await.unwrap();
        
        let data = std::fs::read(dir.path().join("knowledge.json")).unwrap();
        let kb: KnowledgeBase = serde_json::from_slice(&data).unwrap();
        assert_eq!(kb.facts.len(), 1); // Should still be 1 due to deduplication
    }
}
