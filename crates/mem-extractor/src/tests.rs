use super::*;
use mem_core::{Context, MemoryItem, MemoryRole, LlmClient, EmbeddingProvider, MindPalaceConfig, StorageBackend, ImportanceAnalyzer};
use async_trait::async_trait;
use std::sync::Arc;
use serde_json::json;

struct MockLlm {
    response: String,
}

#[async_trait]
impl LlmClient for MockLlm {
    async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
        Ok(self.response.clone())
    }
}

struct MockEmbeddings;

#[async_trait]
impl EmbeddingProvider for MockEmbeddings {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0; 1536])
    }
}

struct MockStorage;
#[async_trait]
impl StorageBackend for MockStorage {
    async fn store(&self, _id: &str, _data: &[u8]) -> anyhow::Result<()> { Ok(()) }
    async fn retrieve(&self, _id: &str) -> anyhow::Result<Vec<u8>> { Ok(vec![]) }
    async fn exists(&self, _id: &str) -> bool { false }
    async fn list(&self, _prefix: &str) -> anyhow::Result<Vec<String>> { Ok(vec![]) }
}

#[tokio::test]
async fn test_extract_facts_robustness() {
    let response = "Certainly! Here are the facts you requested:
```json
[
  {
    \"category\": \"Technical\",
    \"content\": \"Rust is safe\",
    \"confidence\": 0.9,
    \"tags\": [\"programming\"],
    \"dependencies\": [],
    \"scope\": \"Private\"
  }
]
```
I hope this helps!".to_string();

    let extractor: FactExtractor<MockStorage> = FactExtractor::new(
        Arc::new(MockLlm { response }),
        Arc::new(MockEmbeddings),
        MockStorage,
        MindPalaceConfig::default(),
        "knowledge.json".to_string(),
        "test_session".to_string(),
    );

    let context = Context { items: vec![MemoryItem {
        role: MemoryRole::User,
        content: "Tell me about Rust".to_string(),
        timestamp: 0,
        metadata: json!({}),
    }] };

    let facts = extractor.extract_facts(&context).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].content, "Rust is safe");
    assert_eq!(facts[0].category, "Technical");
}

#[tokio::test]
async fn test_score_importance_robustness() {
    let response = "The importance score for this item is 0.85, because it contains critical session data.".to_string();
    
    let extractor: FactExtractor<MockStorage> = FactExtractor::new(
        Arc::new(MockLlm { response }),
        Arc::new(MockEmbeddings),
        MockStorage,
        MindPalaceConfig::default(),
        "knowledge.json".to_string(),
        "test_session".to_string(),
    );

    let item = MemoryItem {
        role: MemoryRole::User,
        content: "Critical info".to_string(),
        timestamp: 0,
        metadata: json!({}),
    };

    let context = Context { items: vec![] };
    let score = ImportanceAnalyzer::score_importance(&extractor, &item, &context).await.unwrap();
    assert!((score - 0.85).abs() < 0.001);
}
