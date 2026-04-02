# MindPalace Development Roadmap

This document outlines the development phases for the 7-layer agent memory crate workspace.

---

## **Phase 0: Foundations & Scaffolding (Current)**
- **Workspace Setup**: Core `Cargo.toml` and crate hierarchy.
- **Core Abstractions**: `mem-core` (Shared traits, `Context`, `MemoryItem`).
- **File-Based Storage**: Implementation of the shared persistence backend.
- **Initial Docs**: `memory_layers.md`, `implementation_spec.md`, `README.md`.

## **Phase 1: Performance & Efficiency (Layers 1-2)**
- **Tool Result Offloader** (`mem-offloader`): Threshold-based chunking.
- **Microcompactor** (`mem-micro`): TTL-based eviction and "in-place" cache preservation.
- **Unit Testing**: Integration between `core` and early optimization layers.

## **Phase 2: Narrative Summarization (Layers 3-4)**
- **Session Summarizer** (`mem-session`): Iterative markdown generation.
- **Full Compactor** (`mem-compactor`): Structural 9-point summary LLM logic.
- **State Management**: Handling summary fallbacks and context restoration.

## **Phase 3: Intelligence & Extraction (Layer 5)**
- **Intent Extraction** (`mem-extractor`): Logic to distill project and user intent.
- **Knowledge Files**: Structured storage for long-term facts.
- **Deduplication**: Ensuring extracted facts don't conflict with active sessions.

## **Phase 4: Consolidation & Dreaming (Layer 6)**
- **Background Task System** (`mem-dreamer`): The Tokio-based background runner.
- **Process Locks**: Robust PID-level file locking.
- **Dreaming Logic**: Summarizing and consolidating across multiple historical sessions.

## **Phase 5: Agent Coordination & Scaling (Layer 7)**
- **Cross-Agent Bridge** (`mem-bridge`): IPC and shared context sharing between agents.
- **CLI Integration**: Exposing memory inspection tools in the `brain` crate.
- **Final Benchmarking**: Validating prompt cache hit rates and token efficiency.

---

## **Milestones Table**

| Milestone | Goal | Status |
| :--- | :--- | :--- |
| **M0** | Documentation & Workspace Scaffolding | **In Progress** |
| **M1** | Working Tool Result Offloading | Pending |
| **M2** | Context window survival over 200k tokens | Pending |
| **M3** | Autonomous Fact Extraction | Pending |
| **M4** | First "Dream" Cycle Success | Pending |
