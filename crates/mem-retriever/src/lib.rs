use mem_core::{Context, FactNode, MemoryItem, MemoryRole, EmbeddingProvider, StorageBackend, KnowledgeBase, LlmClient, FactGraph, utils};
use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
pub use ruvector_core::types::{DistanceMetric, VectorId, HnswConfig};

/// Interface for semantic search and storage of facts in a vector space.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Inserts a fact and its embedding into the store.
    async fn insert(&self, fact: FactNode, embedding: Vec<f32>) -> anyhow::Result<()>;
    /// Searches for the top-k most similar facts for a given query vector.
    async fn search(&self, query: Vec<f32>, top_k: usize, category: Option<String>) -> anyhow::Result<Vec<(FactNode, f32)>>;
    /// Returns all facts CURRENTLY tracked in the store.
    async fn all_facts(&self) -> anyhow::Result<Vec<FactNode>>;
    /// Clears all data from the store for indexing resets.
    async fn clear(&self) -> anyhow::Result<()>;
}

/// A volatile, in-memory implementation of the VectorStore trait.
pub struct InMemoryStore {
    /// Mapping of fact IDs to their node data and semantic embeddings.
    pub facts: RwLock<HashMap<String, (FactNode, Vec<f32>)>>,
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self { facts: RwLock::new(HashMap::new()) }
    }
}

#[async_trait]
impl VectorStore for InMemoryStore {
    async fn insert(&self, fact: FactNode, embedding: Vec<f32>) -> anyhow::Result<()> {
        let mut facts = self.facts.write().await;
        facts.insert(fact.id.clone(), (fact, embedding));
        Ok(())
    }

    async fn search(&self, query: Vec<f32>, top_k: usize, category: Option<String>) -> anyhow::Result<Vec<(FactNode, f32)>> {
        let facts = self.facts.read().await;
        let mut candidates: Vec<_> = facts.values()
            .filter(|(f, _)| {
                category.is_none() || category.as_deref() == Some(&f.category)
            })
            .map(|(fact, embedding)| (fact.clone(), utils::cosine_similarity(&query, embedding)))
            .collect();
        
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(candidates.into_iter().take(top_k).collect())
    }

    async fn all_facts(&self) -> anyhow::Result<Vec<FactNode>> {
        let facts = self.facts.read().await;
        Ok(facts.values().map(|(f, _)| f.clone()).collect())
    }

    async fn clear(&self) -> anyhow::Result<()> {
        let mut facts = self.facts.write().await;
        facts.clear();
        Ok(())
    }
}

/// A persistent, production-grade VectorStore using RuVector HNSW indexing.
///
/// The RuVectorStore manages a dual-storage strategy: 
/// 1. Relational facts are stored in the provided FactGraph (GraphDB).
/// 2. Semantic vectors are stored in the HNSW core index.
pub struct RuVectorStore {
    /// The high-performance semantic index.
    pub index: RwLock<HnswIndex>,
    /// The relational fact graph.
    pub graph: Arc<FactGraph>,
    /// Local cache of vector-to-fact metadata for faster retrieval.
    pub metadata: RwLock<HashMap<VectorId, FactNode>>,
    /// Fixed dimension for incoming embeddings.
    pub dim: usize,
    /// Distance metric (e.g., Cosine, Euclidean) for semantic search.
    pub metric: DistanceMetric,
}

impl RuVectorStore {
    /// Initializes a new RuVectorStore with the specified dimension and metric.
    pub fn new(dim: usize, metric: DistanceMetric, graph: Arc<FactGraph>) -> Self {
        let config = HnswConfig::default();
        Self {
            index: RwLock::new(HnswIndex::new(dim, metric, config).expect("Failed to initialize HnswIndex")),
            graph,
            metadata: RwLock::new(HashMap::new()),
            dim,
            metric,
        }
    }
}

#[async_trait]
impl VectorStore for RuVectorStore {
    /// Inserts a fact into both the relational graph and the semantic vector index.
    async fn insert(&self, fact: FactNode, embedding: Vec<f32>) -> anyhow::Result<()> {
        if embedding.len() != self.dim {
            anyhow::bail!("Embedding dimension mismatch: expected {}, got {}", self.dim, embedding.len());
        }
        
        let id = fact.id.clone();
        
        // Relational storage in FactGraph (GraphDB).
        self.graph.add_fact(fact.clone()).map_err(|e| anyhow::anyhow!("{:?}", e))?;

        // Semantic storage in RuVector-Core (HNSW).
        let mut index = self.index.write().await;
        let mut metadata = self.metadata.write().await;
        
        index.add(id.clone(), embedding).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        metadata.insert(id, fact);
        
        Ok(())
    }

    /// Performs integrated semantic search across vector space and relational graph.
    async fn search(&self, query: Vec<f32>, top_k: usize, category: Option<String>) -> anyhow::Result<Vec<(FactNode, f32)>> {
        let index = self.index.read().await;
        let metadata = self.metadata.read().await;
        
        let results = index.search(&query, top_k).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        
        let mut filtered = Vec::new();
        for result in results {
            if let Some(fact) = metadata.get(&result.id) {
                if let Some(ref cat) = category {
                    if &fact.category != cat { continue; }
                }

                // Integration Policy: Never return superseded facts in semantic results.
                if fact.superseded_by.is_some() { continue; }
                
                filtered.push((fact.clone(), result.score));
            }
        }
        
        Ok(filtered)
    }

    /// Returns a list of all facts indexed in the store.
    async fn all_facts(&self) -> anyhow::Result<Vec<FactNode>> {
        let metadata = self.metadata.read().await;
        Ok(metadata.values().cloned().collect())
    }

    /// Resets the index and metadata cache.
    async fn clear(&self) -> anyhow::Result<()> {
        let mut metadata = self.metadata.write().await;
        metadata.clear();
        let mut index = self.index.write().await;
        *index = HnswIndex::new(self.dim, self.metric, HnswConfig::default()).expect("Failed to reset HnswIndex");
        Ok(())
    }
}

/// Orchestrates semantic memory retrieval and context reconstruction (RAG).
pub struct MemoryRetriever<S: StorageBackend> {
    /// Persistent storage for the base KnowledgeBase.
    pub storage: S,
    /// Provider for query embedding generation.
    pub embeddings: Arc<dyn EmbeddingProvider>,
    /// Client for model calls during context reconstruction.
    pub llm: Arc<dyn LlmClient>,
    /// The specific VectorStore implementation to use (InMemory or RuVector).
    pub store: Arc<dyn VectorStore>,
    /// Access to the relational fact graph.
    pub graph: Arc<FactGraph>,
}

impl<S: StorageBackend> MemoryRetriever<S> {
    /// Initializes a new MemoryRetriever with all required components.
    pub fn new(storage: S, embeddings: Arc<dyn EmbeddingProvider>, llm: Arc<dyn LlmClient>, store: Arc<dyn VectorStore>, graph: Arc<FactGraph>) -> Self {
        Self { storage, embeddings, llm, store, graph }
    }

    /// Factory method for creating a retriever with defaults (InMemory store).
    pub fn legacy(storage: S, embeddings: Arc<dyn EmbeddingProvider>, llm: Arc<dyn LlmClient>) -> Self {
        let graph = Arc::new(FactGraph::new(None).expect("Failed to init fact graph"));
        Self::new(storage, embeddings, llm, Arc::new(InMemoryStore::default()), graph)
    }

    /// Rebuilds the semantic vector index from the persistent KnowledgeBase (Gap 5).
    ///
    /// This method performs mass re-embedding if required, ensuring that the 
    /// volatility of semantic indexes does not result in data loss.
    pub async fn hydrate_from_kb(&self, kb_path: &str) -> anyhow::Result<()> {
        if !self.storage.exists(kb_path).await { return Ok(()); }
        let data = self.storage.retrieve(kb_path).await?;
        if data.is_empty() { return Ok(()); }

        let kb: KnowledgeBase = serde_json::from_slice(&data).unwrap_or_else(|e| {
            tracing::warn!("Failed to parse knowledge base from {}: {}. Starting fresh.", kb_path, e);
            KnowledgeBase::new(None).unwrap_or_default()
        });
        self.store.clear().await?;
        
        for fact in kb.graph.all_active_facts() {
            let embedding = if let Some(ref emb) = fact.embedding {
                emb.clone()
            } else {
                // Perform lazy embedding for facts missing vector data.
                self.embeddings.embed(&fact.content).await?
            };
            self.store.insert(fact, embedding).await?;
        }
        Ok(())
    }

    /// Searches for and retrieves the most relevant facts for the given query.
    pub async fn retrieve_relevant_facts(&self, query: &str, top_k: usize, category_filter: Option<&str>) -> anyhow::Result<Vec<(FactNode, f32)>> {
        let query_embedding = self.embeddings.embed(query).await?;
        self.store.search(query_embedding, top_k, category_filter.map(|s| s.to_string())).await
    }

    /// Reconstructs a high-level agent memory state from all available facts.
    ///
    /// Uses an LLM to synthesize disparate facts into a coherent system 
    /// instruction block for context bootstrap.
    pub async fn bootstrap_context_from_facts(&self) -> anyhow::Result<Context> {
        let facts = self.store.all_facts().await?;
        if facts.is_empty() { return Ok(Context::default()); }
        let prompt = format!("RECONSTRUCT CONTEXT:\n{}", facts.iter().map(|f| format!("- {}", f.content)).collect::<Vec<_>>().join("\n"));
        let reconstructed_state = self.llm.completion(&prompt).await?;
        Ok(Context {
            items: vec![MemoryItem {
                role: MemoryRole::System,
                content: format!("### RECONSTRUCTED AGENT CONTEXT ###\n\n{}", reconstructed_state),
                timestamp: chrono::Utc::now().timestamp() as u64,
                metadata: serde_json::json!({ "bootstrap": true, "fact_count": facts.len() }),
            }],
        })
    }
}
