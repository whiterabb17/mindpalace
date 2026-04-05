# MindPalace Configuration Guide

## Analyzer Timeouts

### Default Values
- **Keyword analyzer**: 500ms (expect <50ms)
- **Heuristic**: 200ms (expect <20ms)
- **Vector**: 8s (expect 1-5s)
- **LLM**: 12s (expect 3-10s)

### Environment Variables
Use standard environment variables to override default timeouts during initialization.
```env
MINDPALACE_KEYWORD_TIMEOUT_MS=500
MINDPALACE_HEURISTIC_TIMEOUT_MS=200
MINDPALACE_VECTOR_TIMEOUT_SECS=8
MINDPALACE_LLM_TIMEOUT_SECS=12
MINDPALACE_FALLBACK_CHAIN_TIMEOUT_SECS=20
MINDPALACE_CIRCUIT_BREAKER_RESET_SECS=45
```

### Tuning Guide
- **Interactive workloads**: Reduce timeouts by 50% for snappier fallbacks down to the Heuristic vector.
- **Background processing**: Increase by 2-3x for exhaustive deep-dives on the LLM layer without dropping tasks.
- **Degradation checks**: If seeing <3 timeouts/min on your metrics dashboard, the current settings are optimal.
