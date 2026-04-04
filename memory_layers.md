# 7-Layer Agent Memory Architecture (Production Hardened)

This document provides a comprehensive technical specification of the 7-layer memory system implemented in the Mentalist agent system. The architecture is designed for **context efficiency**, **zero-copy performance**, and **resilient knowledge retention**.

---

## Layer 0: Shared Context Architecture (`Arc<Context>`)
Before any filtering occurs, the system utilizes an **Atomic Reference Counted** (`Arc`) shared memory model.
- **Goal**: Eliminate expensive cloning of large conversation transcripts across the pipeline.
- **Mechanism**: The `mentalist::Request` struct wraps the `Context` in an `Arc`. Middleware layers clone only the pointer unless a mutation is required, in which case a "copy-on-write" (`*self.state.context).clone()`) pattern is used.

---

## The 7-Layer Defense in Depth

### Layer 1: Tool Result Storage (`mem-offloader`)
- **Goal**: Prevent context "flooding" from verbose tool outputs (e.g., long `cat` results).
- **Mechanism**: If tool output exceeds a configurable threshold (default 2KB), it is persisted to disk. The transcript receives a "Pointer Item" containing a content hash and a 200-character preview.
- **Trigger**: Proactive check during the `after_tool_call` middleware hook.

### Layer 2: Microcompaction (`mem-compactor`)
- **Goal**: Maintain high **Prompt Cache** hit rates by preserving prefix stability.
- **Mechanism**: Silently prunes or summarizes low-value middle items (tool noise, repetitive status) while keeping the system prompt and recent user intent byte-identical.
- **Trigger**: Context usage exceeding 20% of the model's window.

### Layer 3: Session Log (`mem-session`)
- **Goal**: Provide a persistent, human-readable markdown summary of the current session.
- **Mechanism**: The `TodoMiddleware` and `SessionRecorder` maintain a `.agent/session.md` file that maps high-level progress and pending tasks.
- **Trigger**: Every 5 turns or significant milestone detection.

### Layer 4: Deep Summarization (`mem-compactor`)
- **Goal**: Emergency context recovery when hard limits are approached.
- **Mechanism**: A heavyweight `HardLimitCompactor` generates a 9-section structured state report (Goal, Progress, Constraints, etc.) and replaces the bulk of the history with this distilled core.
- **Trigger**: Approaching 80% of the model's hard context limit (e.g., 160k tokens for a 200k model).

### Layer 5: Fact Extraction (`mem-extractor`)
- **Goal**: Durable knowledge retention across disconnected sessions.
- **Mechanism**: Proactively extracts "Project Settings," "User Preferences," and "Technical Facts" into `.agent/knowledge.json`.
- **Trigger**: Executed during `before_ai_call` (RAG retrieval) and `after_ai_call` (New fact discovery). Hardened for streaming via the `run_stream` wrapper.

### Layer 6: Dreaming (`mem-dreamer`)
- **Goal**: Background consolidation and contradiction resolution.
- **Mechanism**: A decoupled background process takes a snapshot of the memory vault, resolves conflicting facts, and optimizes stored knowledge files during idle time.
- **Trigger**: PID-locked background execution when system idle > 5 minutes.

### Layer 7: Cross-Agent Comms (`mem-broker`)
- **Goal**: Efficient context sharing between parent and sub-agents.
- **Mechanism**: Uses the `AgentBridge` to share specialized "Frozen Contexts" with child processes via IPC or JSON-RPC, avoiding full transcript duplication.
- **Trigger**: Parallel task spawning or expert skill delegation.

---

## Design Principles (Hardened)

1. **Byte-Identical Prefixes**: All compaction layers (L2, L4) are designed to preserve the start of the prompt to maximize LLM provider caching.
2. **Atomic Persistence**: Every memory update follows a "Write-to-Temp-then-Rename" pattern to prevent corruption during unexpected crashes.
3. **Resilient Control**: The `ResilientMemoryController` manages all layers, enforcing failure thresholds and guaranteed cleanup of temporary artifacts.
4. **Whitelist-First Safety**: Memory updates from tools are only processed after passing the `CommandValidator` whitelist check.
