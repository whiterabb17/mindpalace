# mem-personality: Personality Guard

![Rust](https://img.shields.io/badge/language-Rust-orange.svg) ![Status: Stable](https://img.shields.io/badge/Status-Stable-green.svg)

`mem-personality` provides a specialized memory layer for maintaining the core identity of an agent in the **MindPalace** memory ecosystem. It ensures that the agent's persona is always present in the context and monitors for "Out of Character" (OOC) drift.

## 🗝️ Key Features

- **Core Identity Enforcement**: Automatically keeps the primary persona prompt at the very beginning of the context (index 0).
- **Immortal Persona**: Marks the persona item with `immortal: true` metadata, protecting it from being pruned by downstream compression or filtering layers.
- **OOC Drift Detection**: Uses a secondary reasoning pass (via an LLM) to verify if the agent's recent responses align with its core identity.
- **Dynamic Re-Alignment**: Injects "Self-Correction" instructions into the hidden context metadata if personality drift is detected, guiding the agent back to its persona in the next response.
- **Priority 1 Execution**: As the first layer in the memory pipeline (Priority 1), it establishes the agent's identity before any other layer processes the context.

## 🏗️ Core Mechanism

The `PersonalityGuard` maintains identity through a dual-pass approach:
1. **Identity Enforcement**: Verifies the persona is at context index 0; if missing, it inserts the persona with "immortal" protection.
2. **Consistency Pass**: Prompts an LLM to review the last assistant response against the core persona and injects re-alignment instructions if necessary.

## 🛠️ Usage Example

```rust
use mem_personality::PersonalityGuard;
use mem_core::{Context, LlmClient, MemoryLayer};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm: Arc<dyn LlmClient> = Arc::new(MyLlmClient::new());
    let persona = "A highly precise technical architect who focuses on Rust performance and safety.".to_string();
    
    let guard = PersonalityGuard::new(persona, Some(llm));
    
    let mut context = Context::default();
    
    // Ensure identity is established and check for character drift
    guard.process(&mut context).await?;
    
    Ok(())
}
```

## 📂 Architecture Context
As **Priority 1**, `mem-personality` is the foundation of every agent turnaround. It ensures that the agent's core reasoning is always anchored in its predefined identity, providing a consistent user experience even in long sessions.
