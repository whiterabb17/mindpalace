use prometheus::{Counter, Gauge, Histogram, HistogramOpts, Opts, Registry};

/// Analyzer-specific metrics (keep Prometheus-native)
pub struct AnalysisMetrics {
    /// Track time spent in each analyzer
    pub llm_analyzer_latency_secs: Histogram,
    pub vector_analyzer_latency_secs: Histogram,
    pub heuristic_analyzer_latency_secs: Histogram,
    pub keyword_analyzer_latency_secs: Histogram,
    
    /// Fallback triggers
    pub llm_timeout_count: Counter,
    pub vector_timeout_count: Counter,
    pub llm_failures_total: Counter,
    pub vector_failures_total: Counter,
    
    /// Score distributions
    pub relevance_scores: Histogram,
    pub importance_scores: Histogram,
    
    /// Cache efficiency
    pub vector_cache_hits: Counter,
    pub vector_cache_misses: Counter,
    
    /// Circuit breaker state
    pub circuit_breaker_state: Gauge,
}

impl AnalysisMetrics {
    pub fn new(registry: &Registry) -> anyhow::Result<Self> {
        let llm_analyzer_latency_secs = Histogram::with_opts(
            HistogramOpts::new(
                "mindpalace_llm_analyzer_latency_seconds",
                "LLM-based analyzer execution time"
            )
            .buckets(vec![0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
        )?;
        registry.register(Box::new(llm_analyzer_latency_secs.clone()))?;

        let vector_analyzer_latency_secs = Histogram::with_opts(
            HistogramOpts::new(
                "mindpalace_vector_analyzer_latency_seconds",
                "Vector embedding analyzer execution time"
            )
            .buckets(vec![0.1, 0.5, 1.0, 3.0, 8.0])
        )?;
        registry.register(Box::new(vector_analyzer_latency_secs.clone()))?;

        let heuristic_analyzer_latency_secs = Histogram::with_opts(
            HistogramOpts::new(
                "mindpalace_heuristic_analyzer_latency_seconds",
                "Heuristic analyzer execution time (fastest)"
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.2])
        )?;
        registry.register(Box::new(heuristic_analyzer_latency_secs.clone()))?;

        let keyword_analyzer_latency_secs = Histogram::with_opts(
            HistogramOpts::new(
                "mindpalace_keyword_analyzer_latency_seconds",
                "Keyword analyzer execution time"
            )
            .buckets(vec![0.001, 0.01, 0.05, 0.1, 0.5])
        )?;
        registry.register(Box::new(keyword_analyzer_latency_secs.clone()))?;

        let llm_timeout_count = Counter::with_opts(Opts::new(
            "mindpalace_llm_analyzer_timeouts_total",
            "Number of LLM analyzer timeouts"
        ))?;
        registry.register(Box::new(llm_timeout_count.clone()))?;

        let vector_timeout_count = Counter::with_opts(Opts::new(
            "mindpalace_vector_analyzer_timeouts_total",
            "Number of vector analyzer timeouts"
        ))?;
        registry.register(Box::new(vector_timeout_count.clone()))?;

        let llm_failures_total = Counter::with_opts(Opts::new(
            "mindpalace_llm_analyzer_failures_total",
            "Total LLM analyzer failures"
        ))?;
        registry.register(Box::new(llm_failures_total.clone()))?;

        let vector_failures_total = Counter::with_opts(Opts::new(
            "mindpalace_vector_analyzer_failures_total",
            "Total vector analyzer failures"
        ))?;
        registry.register(Box::new(vector_failures_total.clone()))?;

        let relevance_scores = Histogram::with_opts(
            HistogramOpts::new(
                "mindpalace_relevance_score",
                "Distribution of relevance scores"
            )
            .buckets(vec![0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        )?;
        registry.register(Box::new(relevance_scores.clone()))?;

        let importance_scores = Histogram::with_opts(
            HistogramOpts::new(
                "mindpalace_importance_score",
                "Distribution of importance scores"
            )
            .buckets(vec![0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        )?;
        registry.register(Box::new(importance_scores.clone()))?;

        let vector_cache_hits = Counter::with_opts(Opts::new(
            "mindpalace_vector_cache_hits_total",
            "Embedding cache hits"
        ))?;
        registry.register(Box::new(vector_cache_hits.clone()))?;

        let vector_cache_misses = Counter::with_opts(Opts::new(
            "mindpalace_vector_cache_misses_total",
            "Embedding cache misses"
        ))?;
        registry.register(Box::new(vector_cache_misses.clone()))?;

        let circuit_breaker_state = Gauge::with_opts(Opts::new(
            "mindpalace_circuit_breaker_state",
            "Circuit breaker state: 0=Closed, 1=Open, 2=HalfOpen"
        ))?;
        registry.register(Box::new(circuit_breaker_state.clone()))?;

        Ok(Self {
            llm_analyzer_latency_secs,
            vector_analyzer_latency_secs,
            heuristic_analyzer_latency_secs,
            keyword_analyzer_latency_secs,
            llm_timeout_count,
            vector_timeout_count,
            llm_failures_total,
            vector_failures_total,
            relevance_scores,
            importance_scores,
            vector_cache_hits,
            vector_cache_misses,
            circuit_breaker_state,
        })
    }
}
