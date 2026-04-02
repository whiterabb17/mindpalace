# 7-Layer Agent Memory Architecture

This document provides a comprehensive analysis of the 7-layer memory system used for sophisticated AI agents (inspired by the architecture of Claude Code). The primary goals of this architecture are **context efficiency**, **cost reduction**, and **long-term knowledge retention**.

---

## Architecture Overview

The system operates on a "Defense in Depth" model for context. Each layer acts as a filter or buffer to prevent the context window from becoming overloaded with redundant or low-value information.

### Layer 1: Tool Result Storage (Offloading)
- **Goal**: Minimize context bloat from verbose tool outputs (e.g., long file reads, network responses).
- **Mechanism**: Tool outputs larger than a threshold (e.g., 2KB) are persisted to local disk. Only a "preview" or "stub" is included in the immediate context.
- **Trigger**: Any tool execution returning more than $N$ characters.

### Layer 2: Microcompaction (Cache Preservation)
- **Goal**: Maintain high "Prompt Cache" hit rates by keeping prefixes byte-identical.
- **Mechanism**: Silently removes old tool results based on their Time-to-Live (TTL) or specific tags. It uses APIs like `cache_edits` to selectively "evict" content from the middle of the transcript without breaking the prefix.
- **Trigger**: Context usage approaching a specific threshold (e.g., 20k tokens).

### Layer 3: Session Memory (Continuous Markdown)
- **Goal**: Provide a cheap, iterative summary of the current session.
- **Mechanism**: A separate, lightweight "summarizer" agent runs concurrently (or at intervals) to maintain a single markdown file mapping the progress of the task.
- **Trigger**: Regular intervals during a long-running session.

### Layer 4: Full Compaction (Deep Summarization)
- **Goal**: Fallback when previous layers fail to keep context under limits.
- **Mechanism**: A heavyweight summarizer agent analyzes the *entire* context and produces a structured 9-section report (covering Goal, Context, Constraints, Progress, etc.). This report replaces the bulk of the recent history.
- **Trigger**: Approaching the hard context limit (~180k-200k tokens).

### Layer 5: Auto Memory Extraction (Durable Knowledge)
- **Goal**: Capture reusable "Facts" that persist across different sessions.
- **Mechanism**: After a session ends or a major milestone is reached, the system extracts "User Intent," "Project Settings," and "Hardcoded Knowledge" into persistent knowledge files (e.g., `.agent/knowledge.json`).
- **Trigger**: Session termination or manual "checkpoint" command.

### Layer 6: Dreaming (Background Consolidation)
- **Goal**: Higher-order reasoning and cleanup during idle time.
- **Mechanism**: A background process (protected by a PID-level mutex lock) reads recent transcripts and knowledge files to consolidate them, resolve contradictions, and optimize stored prompts.
- **Trigger**: System idle time (no user interaction for $X$ minutes).

### Layer 7: Cross-Agent Comms (Coordination)
- **Goal**: Efficient memory sharing between parent and child agents.
- **Mechanism**: Child agents inherit a "frozen" snapshot of the parent's memory. They use specialized messaging protocols to report back to the parent without duplicating the entire context.
- **Trigger**: Spawning sub-tasks or parallel lookups.

---

## Design Principles

1. **Byte-Identical Prefixes**: Always prioritize keeping the start of the prompt exactly the same to ensure Prompt Caching works (crucial for cost).
2. **Cheapest First**: Favor Disk Storage (Layer 1) and Microcompaction (Layer 2) over LLM-driven Summarization (Layers 3 & 4).
3. **Intentional Extraction**: Memory isn't just "everything that happened"; it's the *distilled essence* of what matters for future tasks.
