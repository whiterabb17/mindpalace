use mem_core::{MemoryLayer, Context, LlmClient, FactNode, KnowledgeBase, StorageBackend, EmbeddingProvider, utils, MindPalaceConfig};
use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashSet;

/// Extracts durable facts from conversational context and commits them to the knowledge base.
///
/// The FactExtractor uses an LLM to identify high-level information patterns, categories, 
/// and dependencies within a conversation. It then preserves these in a persistent graph 
/// database with automated semantic deduplication.
pub struct FactExtractor<S: StorageBackend> {
    /// Client for model calls during extraction and reconciliation.
    pub llm: Arc<dyn LlmClient>,
    /// Provider for calculating semantic embeddings of existing and new facts.
    pub embeddings: Arc<dyn EmbeddingProvider>,
    /// Backend for persistent storage of the consolidated KnowledgeBase.
    pub storage: S,
    /// System-wide configuration for thresholds and limits.
    pub config: MindPalaceConfig,
    /// Identifier for the persistent knowledge file.
    pub knowledge_path: String,
    /// Identifier for the current source conversation session.
    pub session_id: String,
}

impl<S: StorageBackend> FactExtractor<S> {
    /// Initializes a new FactExtractor for the specified session.
    pub fn new(llm: Arc<dyn LlmClient>, embeddings: Arc<dyn EmbeddingProvider>, storage: S, config: MindPalaceConfig, knowledge_path: String, session_id: String) -> Self {
        Self { llm, embeddings, storage, config, knowledge_path, session_id }
    }

    /// Identifies and extracts a set of durable JSON-formatted facts from the current context.
    ///
    /// This method performs deep analysis to capture categories, confidence, 
    /// tags, and cross-fact dependencies.
    pub async fn extract_facts(&self, context: &Context) -> anyhow::Result<Vec<FactNode>> {
        let mut history = String::new();
        for item in &context.items { history.push_str(&format!("{:?}: {}\n", item.role, item.content)); }
        let prompt = format!(
"Analyze conversation and extract durable facts as JSON array. 
Fields: 
- category: Technical or personal category
- content: Precise fact
- confidence: 0.0 to 1.0
- tags: Array of strings
- dependencies: Array of content strings this fact relies on
- scope: 'Private' (default), 'Project' (shared in project), or 'Global' (shared ecosystem)

HISTORY:
{}", history);
        let response = self.llm.completion(&prompt).await?;
        let cleaned = response.trim().trim_start_matches("```json").trim_end_matches("```").trim();
        let raw_facts: Vec<serde_json::Value> = serde_json::from_str(cleaned)?;
        Ok(raw_facts.into_iter().filter_map(|v| {
            if let (Some(cat), Some(cont), Some(conf)) = (v["category"].as_str(), v["content"].as_str(), v["confidence"].as_f64()) {
                let mut node = FactNode::new(cont.to_string(), cat.to_string(), conf as f32, self.session_id.clone());
                node.tags = v["tags"].as_array().map_or(Vec::new(), |t| t.iter().filter_map(|s| s.as_str().map(|s| s.to_string())).collect());
                node.dependencies = v["dependencies"].as_array().map_or(Vec::new(), |t| t.iter().filter_map(|s| s.as_str().map(|s| s.to_string())).collect());
                node.scope = match v["scope"].as_str() {
                    Some("Project") => mem_core::FactScope::Project,
                    Some("Global") => mem_core::FactScope::Global,
                    _ => mem_core::FactScope::Private,
                };
                Some(node)
            } else { None }
        }).collect())
    }

    /// Determines if a fact is redundant by checking its semantic distance to existing facts.
    ///
    /// The similarity threshold is determined by the system configuration.
    pub async fn semantic_deduplication(&self, new_fact: &FactNode, existing_facts: &[FactNode]) -> anyhow::Result<bool> {
        let new_embedding = self.embeddings.embed(&new_fact.content).await?;
        for existing in existing_facts {
            let existing_embedding = self.embeddings.embed(&existing.content).await?;
            if utils::cosine_similarity(&new_embedding, &existing_embedding) > self.config.similarity_threshold { 
                return Ok(true); 
            }
        }
        Ok(false)
    }

    /// Integrates newly extracted facts into the KnowledgeBase, resolving conflicts as they arise.
    pub async fn commit_knowledge(&self, new_facts: Vec<FactNode>) -> anyhow::Result<()> {
        let mut kb = if self.storage.exists(&self.knowledge_path).await {
            let data = self.storage.retrieve(&self.knowledge_path).await?;
            serde_json::from_slice(&data).unwrap_or_else(|_| KnowledgeBase::new(None).unwrap())
        } else { KnowledgeBase::new(None)? };

        for fact in new_facts {
            let existing_in_category = kb.graph.query_current(&fact.category);
            // Deduplicate to prevent knowledge base bloat.
            if !self.semantic_deduplication(&fact, &existing_in_category).await? {
                kb.graph.add_fact(fact)?;
            }
        }

        // Detect and resolve contradictions in the updated graph.
        let resolver = ConflictResolver { llm: Arc::clone(&self.llm) };
        resolver.detect_and_resolve_conflicts(&mut kb).await?;
        
        // Persist the updated knowledge base.
        let data = serde_json::to_vec_pretty(&kb)?;
        self.storage.store(&self.knowledge_path, &data).await?;

        Ok(())
    }
}

/// A specialized service for identifying and resolving contradictions within the KnowledgeBase.
pub struct ConflictResolver { 
    /// Client for model calls during conflict arbitration.
    pub llm: Arc<dyn LlmClient>, 
}

/// Strategy for resolving identified contradictions in the fact graph.
pub enum ConflictResolution { 
    /// Preference for the fact with the higher model confidence score.
    KeepHighestConfidence, 
    /// Merges two facts into a third, unified version.
    Merge(FactNode), 
    /// Marks the contradiction for human review or external escalation.
    Escalate 
}

impl ConflictResolver {
    /// Scans the entire KnowledgeBase for category-level contradictions and applies resolutions.
    pub async fn detect_and_resolve_conflicts(&self, kb: &mut KnowledgeBase) -> anyhow::Result<()> {
        let all_facts = kb.graph.all_active_facts();
        let categories: HashSet<_> = all_facts.iter().map(|f| f.category.clone()).collect();

        for category in categories {
            let current_facts = kb.graph.query_current(&category);
            if current_facts.len() > 1 {
                match self.resolve_conflict(current_facts.clone()).await? {
                    ConflictResolution::KeepHighestConfidence => {
                        let best = current_facts.into_iter().max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap()).unwrap();
                        for fact in kb.graph.query_current(&category) {
                            if fact.id != best.id {
                                // Link superseded nodes in the graph instead of deleting them.
                                kb.graph.link_superseded(&fact.id, &best.id)?;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Uses the LLM to determine the appropriate resolution strategy for a set of conflicting facts.
    async fn resolve_conflict(&self, facts: Vec<FactNode>) -> anyhow::Result<ConflictResolution> {
        let prompt = format!("Resolve conflict between facts (KeepHighestConfidence, Merge, Escalate):\n{}", 
            facts.iter().enumerate().map(|(i, f)| format!("{}. [conf: {}] {}", i+1, f.confidence, f.content)).collect::<Vec<_>>().join("\n"));
        let res = self.llm.completion(&prompt).await?;
        if res.contains("KeepHighestConfidence") { Ok(ConflictResolution::KeepHighestConfidence) }
        else if res.contains("Merge") { Ok(ConflictResolution::KeepHighestConfidence) }
        else { Ok(ConflictResolution::Escalate) }
    }
}

#[async_trait]
impl<S: StorageBackend> MemoryLayer for FactExtractor<S> {
    fn name(&self) -> &str { "FactExtractor" }

    /// Executes the extraction and commitment logic if a "milestone" is flagged in metadata.
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        if context.items.last().map_or(false, |item| item.metadata["milestone"].as_bool().unwrap_or(false)) {
            let facts = self.extract_facts(context).await?;
            self.commit_knowledge(facts).await?;
        }
        Ok(())
    }

    /// Lower priority ensures this runs after summary or filtering layers.
    fn priority(&self) -> u32 { 5 }
}

/// A secondary layer that triggers fact extraction based on specific conversation triggers.
pub struct ReflectionLayer<S: StorageBackend> { 
    /// The primary fact extractor service.
    pub extractor: Arc<FactExtractor<S>>, 
}

impl<S: StorageBackend> ReflectionLayer<S> { pub fn new(extractor: Arc<FactExtractor<S>>) -> Self { Self { extractor } } }

#[async_trait]
impl<S: StorageBackend> MemoryLayer for ReflectionLayer<S> {
    fn name(&self) -> &str { "ReflectionLayer" }

    /// Triggers extraction if the user explicitly corrects a fact or asks to remember something.
    async fn process(&self, context: &mut Context) -> anyhow::Result<()> {
        if context.items.len() < 2 { return Ok(()); }
        let last_content = context.items.last().unwrap().content.to_lowercase();
        if last_content.contains("remember") || last_content.contains("fact:") || last_content.contains("actually,") {
            let facts = self.extractor.extract_facts(context).await?;
            self.extractor.commit_knowledge(facts).await?;
        }
        Ok(())
    }

    /// High priority ensures this captures the context before any pruning or summarization.
    fn priority(&self) -> u32 { 4 }
}
