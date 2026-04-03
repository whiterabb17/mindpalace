use mem_core::{MemoryLayer, Context, MemoryRole, MemoryItem, LlmClient};
use async_trait::async_trait;
use std::sync::Arc;

/// A specialized memory layer for enforcing persona consistency and role-play maintenance.
///
/// The PersonalityGuard ensures that the system's core identity (the persona) 
/// is always present in the context and monitors for "Out of Character" (OOC) drift 
/// using a secondary reasoning pass.
pub struct PersonalityGuard {
    /// The primary persona prompt as defined in the system's root configuration.
    pub core_persona: String,
    /// Optional LLM client for automated OOC detection and re-alignment.
    pub llm: Option<Arc<dyn LlmClient>>,
}

impl PersonalityGuard {
    /// Initializes a new PersonalityGuard with the specified persona content.
    pub fn new(core_persona: String, llm: Option<Arc<dyn LlmClient>>) -> Self {
        Self { core_persona, llm }
    }

    /// Verifies that the agent's recent responses align with the core persona.
    ///
    /// If drift is detected, it injects a "Self-Correction" instruction into 
    /// the hidden metadata for the next model turn.
    pub async fn check_consistency(&self, context: &mut Context) -> anyhow::Result<()> {
        let Some(llm) = &self.llm else { return Ok(()); };
        
        let last_assistant_msg = context.items.iter().rev()
            .find(|i| i.role == MemoryRole::Assistant)
            .map(|i| i.content.clone());
            
        if let Some(content) = last_assistant_msg {
            let prompt = format!(
"Evaluate if the following response aligns with the PERSONA. 
Reply with 'PASS' if consistent, or a brief 'RE-ALIGN' instruction if inconsistent.

PERSONA: {}
RESPONSE: {}
", self.core_persona, content);

            let review = llm.completion(&prompt).await?;
            if review.contains("RE-ALIGN") {
                tracing::warn!("Persona drift detected. Injecting re-alignment metadata.");
                if let Some(last) = context.items.last_mut() {
                    last.metadata["persona_correction"] = serde_json::Value::String(review);
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl MemoryLayer for PersonalityGuard {
    fn name(&self) -> &str { "PersonalityGuard" }

    /// Ensures the System persona is at the start and checks for recent drift.
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        // 1. Core Identity Enforcement (Hard anchor at index 0)
        let has_persona = context.items.first()
            .map_or(false, |i| i.role == MemoryRole::System && i.content.contains(&self.core_persona));
            
        if !has_persona {
            context.items.insert(0, MemoryItem {
                role: MemoryRole::System,
                content: format!("CORE PERSONA: {}", self.core_persona),
                timestamp: chrono::Utc::now().timestamp() as u64,
                metadata: serde_json::from_str(r#"{"immortal": true}"#).unwrap(),
            });
        }

        // 2. Consistency Pass (if LLM is available)
        self.check_consistency(context).await?;

        Ok(())
    }

    /// Priority 1 ensures the identity is established before any other layer processes the context.
    fn priority(&self) -> u32 { 1 }
}
