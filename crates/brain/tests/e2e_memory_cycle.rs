use brain::Brain;
use mem_core::{Context, MemoryItem, MemoryRole, MindPalaceConfig, FileStorage, StorageBackend, LlmClient, EmbeddingProvider, ImportanceAnalyzer};
use mem_extractor::{FactExtractor, ReflectionLayer};
use mem_session::SessionSummarizer;
use mem_micro::{AdaptiveMicroCompactor, TTLDecayStrategy};
use mem_core::analysis::{HeuristicImportanceAnalyzer, KeywordRelevanceAnalyzer};
use std::sync::Arc;
use tempfile::tempdir;
use async_trait::async_trait;

// --- Production-Ready Test Implementation (NO MOCKS in logic) ---

struct TestLlm;
#[async_trait]
impl LlmClient for TestLlm {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String> {
        if prompt.to_lowercase().contains("extract") && prompt.to_lowercase().contains("facts") {
            Ok(r#"[{"category": "Technical", "content": "The database uses PostgreSQL version 14.", "confidence": 0.95, "tags": ["db", "v14"], "dependencies": []}]"#.to_string())
        } else if prompt.to_lowercase().contains("markdown summary") {
            Ok("Summary of the session logic.".to_string())
        } else if prompt.contains("PASS") || prompt.contains("FAIL") {
            Ok("PASS".to_string())
        } else if prompt.to_lowercase().contains("structural") {
            Ok("1. GOAL: Testing. 2. CONTEXT: None. 3. TOOLS: None. 4. ERRORS: None. 5. PROGRESS: 100%. 6. PENDING: None. 7. CONSTANTS: None. 8. PREFERENCES: None. 9. NEXT: Exit.".to_string())
        } else {
            Ok("0.8".to_string()) // Default numeric response for heuristic fallback if needed
        }
    }
}

struct TestEmbeddings;
#[async_trait]
impl EmbeddingProvider for TestEmbeddings {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.1; 128])
    }
}

#[tokio::test]
async fn test_e2e_memory_pipeline_cycle() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt().try_init();

    let dir = tempdir()?;
    let storage = FileStorage::new(dir.path().to_path_buf());
    let config = MindPalaceConfig {
        similarity_threshold: 0.85,
        compression_ratio: 0.5,
        max_context_items: 5, 
        base_ttl_seconds: 3600,
        idle_threshold_mins: 45,
        summary_interval: 3,
        default_model: "test".to_string(),
        max_tokens_per_dream: 50_000,
        retention_sessions: 10,
        model_context_window: 2048,
    };

    let llm: Arc<dyn LlmClient> = Arc::new(TestLlm);
    let embeddings: Arc<dyn EmbeddingProvider> = Arc::new(TestEmbeddings);
    let mut brain = Brain::new(config.clone(), None, None);

    // 1. Setup Layers
    let extractor = Arc::new(FactExtractor::new(
        Arc::clone(&llm),
        Arc::clone(&embeddings),
        storage.clone(),
        config.clone(),
        "knowledge.json".to_string(),
        "test_session".to_string(),
    ));
    brain.add_layer(Arc::new(ReflectionLayer::new(Arc::clone(&extractor))));

    brain.add_layer(Arc::new(SessionSummarizer::new(
        Arc::clone(&llm),
        storage.clone(),
        config.clone(),
        "narratives".to_string(),
        true,
    )));

    // Use PRODUCTION Heuristic Analyzers from the library instead of local mocks.
    brain.add_layer(Arc::new(AdaptiveMicroCompactor::new(
        config.clone(),
        TTLDecayStrategy::AdaptiveByType,
        Arc::new(KeywordRelevanceAnalyzer),
    )));

    // --- EXECUTION CYCLE ---

    let mut context = Context::default();
    let now = chrono::Utc::now().timestamp() as u64;
    
    // START with System Prompt
    context.items.push(MemoryItem {
        role: MemoryRole::System,
        content: "You are a helpful assistant.".to_string(),
        timestamp: now - 100,
        metadata: serde_json::json!({}),
    });

    // Step 1: Fact Extraction via Reflection
    context.items.push(MemoryItem {
        role: MemoryRole::User,
        content: "Actually, we use Postgres v14.".to_string(),
        timestamp: now - 50,
        metadata: serde_json::json!({}),
    });

    // Run optimization
    brain.optimize(&mut context).await?;

    // Verify facts were extracted
    assert!(storage.exists("knowledge.json").await);

    // Step 2: Session Summarization
    context.items.push(MemoryItem {
        role: MemoryRole::Assistant,
        content: "I'll make a note of that.".to_string(),
        timestamp: now - 40,
        metadata: serde_json::json!({}),
    });

    brain.optimize(&mut context).await?;

    // Verify summarization happened
    assert!(context.items.len() <= 3);
    assert!(context.items[0].content.contains("SESSION NARRATIVE SUMMARY"));

    // Step 3: Hard Limit Protection (max_context_items = 5)
    for i in 0..10 {
        context.items.push(MemoryItem {
            role: MemoryRole::User,
            content: format!("Message {}", i),
            timestamp: now + i as u64,
            metadata: serde_json::json!({}),
        });
    }

    brain.optimize(&mut context).await?;

    // Hard limit should have pruned it down to exactly 5 items
    assert_eq!(context.items.len(), 5);

    // Step 4: Verify Heuristic Importance Scoring (Logic check)
    let heuristic = HeuristicImportanceAnalyzer;
    let important_item = MemoryItem {
        role: MemoryRole::User,
        content: "This is a critical decision about the fact of the architecture.".to_string(),
        timestamp: now,
        metadata: serde_json::json!({}),
    };
    let score = heuristic.score_importance(&important_item, &context).await?;
    assert!(score > 0.7, "Heuristic should score important architectural keywords highly.");

    Ok(())
}
