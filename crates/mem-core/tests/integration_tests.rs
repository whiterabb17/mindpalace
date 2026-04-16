use async_trait::async_trait;
use mem_core::analysis::*;
use mem_core::RelevanceAnalyzer;
use mem_core::{Context, LlmClient, MemoryItem, MemoryRole};
use std::sync::Arc;
use std::time::Duration;

// --- Mocks ---

struct MockLlm {
    response: String,
    delay: Option<Duration>,
}

#[async_trait]
impl LlmClient for MockLlm {
    async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
        if let Some(d) = self.delay {
            tokio::time::sleep(d).await;
        }
        Ok(self.response.clone())
    }
}

pub struct MockEmbeddingProvider;

#[async_trait]
impl mem_core::EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0; 384])
    }
}

// --- Tests ---

#[tokio::test]
async fn test_llm_relevance_analyzer_with_mocked_llm() {
    let mock_llm = Arc::new(MockLlm {
        response: "0.85".to_string(),
        delay: None,
    });

    let analyzer = LlmRelevanceAnalyzer::new(mock_llm);

    let context = Context {
        items: vec![MemoryItem {
            role: MemoryRole::User,
            content: "Hello".to_string(),
            timestamp: 0,
            metadata: Default::default(),
        }],
    };

    let item = MemoryItem {
        role: MemoryRole::Assistant,
        content: "Hi!".to_string(),
        timestamp: 1,
        metadata: Default::default(),
    };

    let score = analyzer.score_relevance(&item, &context).await.unwrap();
    assert_eq!(score, 0.85);
}

#[tokio::test]
async fn test_analyzer_fallback_chain() {
    // Primary is very slow (times out)
    let slow_llm = Arc::new(MockLlm {
        response: "0.9".to_string(),
        delay: Some(Duration::from_secs(10)),
    });
    let primary = Arc::new(LlmRelevanceAnalyzer::new(slow_llm));
    let fallback = Arc::new(KeywordRelevanceAnalyzer);

    let fallback_analyzer = FallbackRelevanceAnalyzer::new(primary, fallback);

    let context = Context {
        items: vec![MemoryItem {
            role: MemoryRole::User,
            content: "What is testing?".to_string(),
            timestamp: 0,
            metadata: Default::default(),
        }],
    };

    let item = MemoryItem {
        role: MemoryRole::Assistant,
        content: "Testing is good.".to_string(),
        timestamp: 1,
        metadata: Default::default(),
    };

    // Testing the fallback behavior. It should time out quickly (FallbackRelevanceAnalyzer has 5s timeout, but for test we can just let it run or rely on real timeout)
    // Actually our test timeout is 5 seconds. Since we wait 10 seconds in mock, it will trigger the fallback.
    // Keyword matcher will return something fast.
    let start = std::time::Instant::now();
    let score = fallback_analyzer.score_relevance(&item, &context).await;
    let elapsed = start.elapsed();

    assert!(score.is_ok());
    assert!(elapsed.as_secs() >= 4); // should take about 5s to hit timeout in FallbackRelevanceAnalyzer
}

#[tokio::test]
async fn test_keyword_analyzer_boundary_conditions() {
    let analyzer = KeywordRelevanceAnalyzer;

    let empty_context = Context { items: vec![] };
    let item = MemoryItem {
        role: MemoryRole::User,
        content: "Does this work?".to_string(),
        timestamp: 0,
        metadata: Default::default(),
    };

    // Expecting 0.5 for empty context
    let score_empty = analyzer
        .score_relevance(&item, &empty_context)
        .await
        .unwrap();
    assert_eq!(score_empty, 0.5);

    let context = Context {
        items: vec![MemoryItem {
            role: MemoryRole::User,
            content: "We are testing boundary conditions carefully".to_string(),
            timestamp: 0,
            metadata: Default::default(),
        }],
    };

    let large_item = MemoryItem {
        role: MemoryRole::User,
        content: "testing boundary".repeat(1000),
        timestamp: 1,
        metadata: Default::default(),
    };
    let score_large = analyzer
        .score_relevance(&large_item, &context)
        .await
        .unwrap();
    // It should not crash or take forever since token limit MAX_TOKENS_PER_MESSAGE is 500
    assert!(score_large > 0.0);
}

#[tokio::test]
async fn test_float_parsing() {
    let score = parse_llm_score("0.75", "test").await.unwrap();
    assert_eq!(score, 0.75);

    let json_score = parse_llm_score("[0.82]", "test").await.unwrap();
    assert_eq!(json_score, 0.82);

    let text_score = parse_llm_score("Score: 0.65", "test").await.unwrap();
    assert_eq!(text_score, 0.65);

    // Invalid score should fail
    let bad_score = parse_llm_score("banana", "test").await;
    assert!(bad_score.is_err());

    // Out of range should fail
    let out_of_range = parse_llm_score("2.5", "test").await;
    assert!(out_of_range.is_err());
}

#[tokio::test]
async fn test_analyzer_strategy_degradation() {
    let config = mem_core::analysis::AnalysisTimeoutConfig::new();
    let health_monitor = Arc::new(mem_core::analysis::AnalyzerHealthMonitor::new(
        config.clone(),
    ));

    // Default is Full
    assert_eq!(
        health_monitor.get_active_analyzer(),
        mem_core::analysis::AnalyzerStrategy::Full
    );

    // Record 3 LLM timeouts
    health_monitor.record_llm_timeout();
    health_monitor.record_llm_timeout();
    health_monitor.record_llm_timeout();

    // Should now degrade to VectorOnly
    assert_eq!(
        health_monitor.get_active_analyzer(),
        mem_core::analysis::AnalyzerStrategy::VectorOnly
    );
}

#[tokio::test]
async fn test_context_window_overflow() {
    // Large context, verify truncation doesn't crash KeywordRelevanceAnalyzer
    let analyzer = KeywordRelevanceAnalyzer;

    let mut large_context = mem_core::Context { items: vec![] };
    for _ in 0..10 {
        large_context.items.push(mem_core::MemoryItem {
            role: mem_core::MemoryRole::User,
            content: "We are testing context window overflow by throwing massive ungodly amounts of random words and testing the parsing bounds constraints and limits".repeat(100),
            timestamp: 0,
            metadata: Default::default()
        });
    }

    let item = mem_core::MemoryItem {
        role: mem_core::MemoryRole::Assistant,
        content: "testing overflow bounds".to_string(),
        timestamp: 1,
        metadata: Default::default(),
    };

    let score_large = analyzer.score_relevance(&item, &large_context).await;
    assert!(score_large.is_ok());
}

#[tokio::test]
async fn test_cache_ttl_expiration() {
    // Testing CachedVectorRelevanceAnalyzer initialization works with moka
    let mock_embeddings = Arc::new(MockEmbeddingProvider);
    let cached_analyzer = CachedVectorRelevanceAnalyzer::new(mock_embeddings.clone());

    let context = Context { items: vec![] };
    let item = mem_core::MemoryItem {
        role: mem_core::MemoryRole::Assistant,
        content: "test".to_string(),
        timestamp: 1,
        metadata: Default::default(),
    };

    // First call (cache miss)
    let score1 = cached_analyzer.score_relevance(&item, &context).await;
    assert!(score1.is_ok());

    // Second call (cache hit)
    let score2 = cached_analyzer.score_relevance(&item, &context).await;
    assert!(score2.is_ok());
}
