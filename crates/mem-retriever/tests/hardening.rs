use async_trait::async_trait;
use mem_core::{EmbeddingProvider, FactGraph, FileStorage, LlmClient};
use mem_retriever::{DistanceMetric, MemoryRetriever, RuVectorStore};
use std::sync::Arc;
use tempfile::tempdir;

struct MockEmbeddings;
#[async_trait]
impl EmbeddingProvider for MockEmbeddings {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0; 1536])
    }
}

struct MockLlm;
#[async_trait]
impl LlmClient for MockLlm {
    async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
        Ok("[]".to_string())
    }
}

#[tokio::test]
async fn test_hydrate_from_empty_kb() {
    let dir = tempdir().unwrap();
    let storage = FileStorage::new(dir.path().to_path_buf());
    let embeddings = Arc::new(MockEmbeddings);
    let llm = Arc::new(MockLlm);
    let graph = Arc::new(FactGraph::new(None).unwrap());
    let store = Arc::new(RuVectorStore::new(
        1536,
        DistanceMetric::Cosine,
        graph.clone(),
    ));
    let retriever = MemoryRetriever::new(storage.clone(), embeddings, llm, store, graph);

    // Create an empty file
    let kb_path = "knowledge.json";
    tokio::fs::write(dir.path().join(kb_path), b"")
        .await
        .unwrap();

    // This should NOT crash with "expected value at line 1 column 1"
    retriever.hydrate_from_kb(kb_path).await.unwrap();
}

#[tokio::test]
async fn test_hydrate_from_corrupt_kb() {
    let dir = tempdir().unwrap();
    let storage = FileStorage::new(dir.path().to_path_buf());
    let embeddings = Arc::new(MockEmbeddings);
    let llm = Arc::new(MockLlm);
    let graph = Arc::new(FactGraph::new(None).unwrap());
    let store = Arc::new(RuVectorStore::new(
        1536,
        DistanceMetric::Cosine,
        graph.clone(),
    ));
    let retriever = MemoryRetriever::new(storage.clone(), embeddings, llm, store, graph);

    // Create a corrupt JSON file
    let kb_path = "knowledge.json";
    tokio::fs::write(dir.path().join(kb_path), b"{ invalid json }")
        .await
        .unwrap();

    // This should NOT crash, but log a warning and return Ok(())
    retriever.hydrate_from_kb(kb_path).await.unwrap();
}
