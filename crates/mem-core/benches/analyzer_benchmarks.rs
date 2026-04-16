use criterion::{criterion_group, criterion_main, Criterion};
use mem_core::analysis::*;
use mem_core::{Context, ImportanceAnalyzer, RelevanceAnalyzer};
use std::hint::black_box;

fn bench_keyword_analyzer(c: &mut Criterion) {
    let analyzer = KeywordRelevanceAnalyzer;

    let mut large_context = mem_core::Context { items: vec![] };
    for _ in 0..5 {
        large_context.items.push(mem_core::MemoryItem {
            role: mem_core::MemoryRole::User,
            content: "We are testing context window overflow by throwing massive ungodly amounts of random words and testing the parsing bounds constraints and limits ".repeat(20),
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

    c.bench_function("keyword_relevance_large_context", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                analyzer
                    .score_relevance(black_box(&item), black_box(&large_context))
                    .await
                    .unwrap()
            })
    });
}

fn bench_heuristic_importance(c: &mut Criterion) {
    let analyzer = HeuristicImportanceAnalyzer;
    let item = mem_core::MemoryItem {
        role: mem_core::MemoryRole::System,
        content: "Goal: Implement a highly scalable architecture and fix technical debt."
            .to_string(),
        timestamp: 1,
        metadata: Default::default(),
    };
    let context = Context { items: vec![] };

    c.bench_function("heuristic_importance_regex", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                analyzer
                    .score_importance(black_box(&item), black_box(&context))
                    .await
                    .unwrap()
            })
    });
}

criterion_group!(benches, bench_keyword_analyzer, bench_heuristic_importance);
criterion_main!(benches);
