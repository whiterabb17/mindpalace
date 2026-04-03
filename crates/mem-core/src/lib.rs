use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::PathBuf;
use uuid::Uuid;
use prometheus::{Histogram, Gauge, Registry, opts, register_histogram_with_registry, register_gauge_with_registry};
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use ruvector_graph::{GraphDB, Node, Edge, Label, Properties, PropertyValue};

/// Represents the role of a participant in a conversation or memory event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryRole { 
    /// The user interacting with the agent.
    User, 
    /// The AI agent response.
    Assistant, 
    /// Output from a tool execution.
    Tool, 
    /// System-level instructions or status updates.
    System 
}

/// A discrete unit of knowledge within the FactGraph, enriched with relationships and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactNode {
    /// Unique identifier for the fact.
    pub id: String,
    /// The factual content as extracted by an LLM.
    pub content: String,
    /// Broad category for organization (e.g., "Personal", "Technical").
    pub category: String,
    /// Confidence score of the extraction (0.0 to 1.0).
    pub confidence: f32,
    /// Unix timestamp of creation.
    pub timestamp: u64,
    /// Incremental version of the fact as it evolves.
    pub version: u32,
    /// Optional ID of a newer fact that supersedes this one.
    pub superseded_by: Option<String>,
    /// List of fact IDs that this fact depends on.
    pub dependencies: Vec<String>,
    /// Optional timestamp for when this fact is no longer valid.
    pub valid_until: Option<u64>,
    /// The session ID where this fact was originally extracted.
    pub source_session_id: String,
    /// User-defined or AI-generated tags for searchability.
    pub tags: Vec<String>,
    /// Semantic vector representation for RAG retrieval.
    pub embedding: Option<Vec<f32>>,
}

impl FactNode {
    /// Creates a new FactNode with the given content and category.
    pub fn new(content: String, category: String, confidence: f32, source_session_id: String) -> Self {
        Self { 
            id: Uuid::new_v4().to_string(), 
            content, 
            category, 
            confidence, 
            timestamp: chrono::Utc::now().timestamp() as u64, 
            version: 1, 
            superseded_by: None, 
            dependencies: Vec::new(), 
            valid_until: None, 
            source_session_id, 
            tags: Vec::new(), 
            embedding: None 
        }
    }

    /// Converts the node into property-value pairs for GraphDB storage.
    pub fn to_properties(&self) -> Properties {
        let mut props = Properties::new();
        props.insert("content".into(), PropertyValue::from(self.content.clone()));
        props.insert("category".into(), PropertyValue::from(self.category.clone()));
        props.insert("confidence".into(), PropertyValue::from(self.confidence as f64));
        props.insert("timestamp".into(), PropertyValue::from(self.timestamp as i64));
        props.insert("version".into(), PropertyValue::from(self.version as i32));
        props.insert("source_session_id".into(), PropertyValue::from(self.source_session_id.clone()));
        if let Some(ref emb) = self.embedding {
            props.insert("embedding".into(), PropertyValue::from(emb.clone()));
        }
        props
    }

    /// Reconstructs a FactNode from GraphDB properties.
    pub fn from_properties(id: String, props: &Properties) -> Self {
        let get_str = |key: &str| {
            props.get(key).and_then(|v| serde_json::to_value(v).ok())
                .and_then(|jv| if jv.is_string() { jv.as_str().map(|s| s.to_string()) } else { Some(jv.to_string()) })
                .unwrap_or_default()
        };
        let get_f32 = |key: &str| {
            props.get(key).and_then(|v| serde_json::to_value(v).ok())
                .and_then(|jv| jv.as_f64()).unwrap_or(0.0) as f32
        };
        let get_u64 = |key: &str| {
            props.get(key).and_then(|v| serde_json::to_value(v).ok())
                .and_then(|jv| jv.as_u64()).unwrap_or(0)
        };
        let get_u32 = |key: &str| {
            props.get(key).and_then(|v| serde_json::to_value(v).ok())
                .and_then(|jv| jv.as_u64()).unwrap_or(1) as u32
        };

        Self {
            id,
            content: get_str("content"),
            category: get_str("category"),
            confidence: get_f32("confidence"),
            timestamp: get_u64("timestamp"),
            version: get_u32("version"),
            superseded_by: None,
            dependencies: Vec::new(),
            valid_until: None,
            source_session_id: get_str("source_session_id"),
            tags: Vec::new(),
            embedding: None,
        }
    }
}

/// A wrap around ruvector-graph::GraphDB optimized for conversational fact management.
pub struct FactGraph { 
    /// The underlying graph database instance.
    pub db: Arc<GraphDB>,
    /// Tracks all managed node IDs to support iteration and reconciliation across sessions.
    pub node_ids: std::sync::RwLock<std::collections::HashSet<String>>,
}

impl Serialize for FactGraph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {
        let ids = self.node_ids.read().unwrap();
        let ids_vec: Vec<_> = ids.iter().cloned().collect();
        ids_vec.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FactGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: serde::Deserializer<'de> {
        let ids_vec = Vec::<String>::deserialize(deserializer)?;
        let graph = FactGraph::new(None).map_err(serde::de::Error::custom)?;
        {
            let mut ids = graph.node_ids.write().unwrap();
            for id in ids_vec { ids.insert(id); }
        }
        Ok(graph)
    }
}

impl FactGraph {
    /// Initializes a new FactGraph, optionally loading from a persistence path.
    pub fn new(path: Option<PathBuf>) -> anyhow::Result<Self> {
        let db = if let Some(p) = path {
            GraphDB::with_storage(p).map_err(|e| anyhow::anyhow!("{:?}", e))?
        } else {
            GraphDB::new() 
        };
        Ok(Self { 
            db: Arc::new(db),
            node_ids: std::sync::RwLock::new(std::collections::HashSet::new()),
        })
    }

    /// Retrieves a single fact by its unique ID.
    pub fn get_fact(&self, id: &str) -> Option<FactNode> { 
        self.db.get_node(id).map(|node| FactNode::from_properties(node.id.clone(), &node.properties))
    }

    /// Returns all current (non-superseded) facts within a specific category.
    pub fn query_current(&self, category: &str) -> Vec<FactNode> { 
        let ids = self.node_ids.read().unwrap();
        ids.iter()
            .filter_map(|id| self.get_fact(id))
            .filter(|f| f.category == category && f.superseded_by.is_none())
            .collect()
    }

    /// Returns all facts that have not been superseded by newer information.
    pub fn all_active_facts(&self) -> Vec<FactNode> { 
        let ids = self.node_ids.read().unwrap();
        ids.iter()
            .filter_map(|id| self.get_fact(id))
            .filter(|f| f.superseded_by.is_none())
            .collect()
    }

    /// Inserts a new fact into the graph and adds its ID to the identity set.
    pub fn add_fact(&self, fact: FactNode) -> anyhow::Result<()> {
        let node = Node::new(fact.id.clone(), vec![Label::new("Fact")], fact.to_properties());
        {
            let mut ids = self.node_ids.write().unwrap();
            ids.insert(fact.id.clone());
        }
        self.db.create_node(node).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok(())
    }

    /// Creates a directional link marking an old fact as superseded by a newer one.
    pub fn link_superseded(&self, old_id: &str, new_id: &str) -> anyhow::Result<()> {
        let edge = Edge::new(Uuid::new_v4().to_string(), old_id.to_string(), new_id.to_string(), "SUPERSEDED_BY".into(), Properties::new());
        self.db.create_edge(edge).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok(())
    }

    /// Creates a directional link indicating that one fact relies on another.
    pub fn link_dependency(&self, source_id: &str, target_id: &str) -> anyhow::Result<()> {
        let edge = Edge::new(Uuid::new_v4().to_string(), source_id.to_string(), target_id.to_string(), "DEPENDS_ON".into(), Properties::new());
        self.db.create_edge(edge).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok(())
    }
}

/// A trait for calculating the token footprint of strings across different model providers.
pub trait TokenCounter: Send + Sync { 
    /// Returns the exact count of tokens in the given text.
    fn count_tokens(&self, text: &str) -> usize; 
}

/// A provider that generates vector embeddings for text suitable for semantic retrieval.
#[async_trait] 
pub trait EmbeddingProvider: Send + Sync { 
    /// Generates a vector representation for the specified input string.
    async fn embed(&self, text: &str) -> anyhow::Result<Vec< f32 >>; 
}

/// The top-level knowledge container aggregating the fact graph and metadata.
#[derive(Serialize, Deserialize)]
pub struct KnowledgeBase { 
    /// The relational fact graph.
    pub graph: FactGraph, 
}

impl KnowledgeBase { 
    /// Initializes a new KnowledgeBase at the specified location.
    pub fn new(path: Option<PathBuf>) -> anyhow::Result<Self> { Ok(Self { graph: FactGraph::new(path)? }) } 
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new(None).expect("Failed to create default knowledge base")
    }
}

/// A single message or event in the agent's short-term context.
#[derive(Debug, Clone, Serialize, Deserialize)] 
pub struct MemoryItem { 
    /// Role of the sender (e.g., User, Assistant).
    pub role: MemoryRole, 
    /// Content of the message.
    pub content: String, 
    /// Unix timestamp of the event.
    pub timestamp: u64, 
    /// Extensible metadata for layer processing (e.g., importance scores, TTL).
    pub metadata: serde_json::Value, 
}

/// The total short-term memory state of the agent.
#[derive(Debug, Default, Clone, Serialize, Deserialize)] 
pub struct Context { 
    /// List of ordered memory items.
    pub items: Vec<MemoryItem>, 
}

/// A specialized logic layer that transforms or prunes the agent's context.
#[async_trait] 
pub trait MemoryLayer: Send + Sync { 
    /// Unique name of the layer for instrumentation.
    fn name(&self) -> &str; 
    /// Processes and modifies the current context.
    async fn process(&self, context: &mut Context) -> anyhow::Result<()>; 
    /// Determines execution order (lower is earlier).
    fn priority(&self) -> u32; 
}

/// A generic interface for persistent storage (Disk, S3, Redis, etc.)
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Stores binary data with the specified identifier.
    async fn store(&self, id: &str, data: &[u8]) -> anyhow::Result<()>;
    /// Retrieves binary data by identifier.
    async fn retrieve(&self, id: &str) -> anyhow::Result<Vec<u8>>;
    /// Checks for existence of an identifier.
    async fn exists(&self, id: &str) -> bool;
    /// Lists all identifiers with a matching prefix.
    async fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>>;
}

/// A local file system implementation of the StorageBackend.
#[derive(Clone)] pub struct FileStorage { 
    /// The root path for data storage.
    pub root: PathBuf, 
}
impl FileStorage { pub fn new(root: PathBuf) -> Self { Self { root } } }
#[async_trait]
impl StorageBackend for FileStorage {
    async fn store(&self, id: &str, data: &[u8]) -> anyhow::Result<()> {
        let path = self.root.join(id);
        if let Some(parent) = path.parent() { tokio::fs::create_dir_all(parent).await?; }
        tokio::fs::write(path, data).await?;
        Ok(())
    }
    async fn retrieve(&self, id: &str) -> anyhow::Result<Vec<u8>> { let path = self.root.join(id); Ok(tokio::fs::read(path).await?) }
    async fn exists(&self, id: &str) -> bool { self.root.join(id).exists() }
    async fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>> {
        let scan_path = self.root.join(prefix);
        if !scan_path.exists() { return Ok(Vec::new()); }
        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(scan_path).await?;
        while let Some(entry) = read_dir.next_entry().await? { if let Some(name) = entry.file_name().to_str() { entries.push(name.to_string()); } }
        Ok(entries)
    }
}

/// High-level client for executing model completions.
#[async_trait] pub trait LlmClient: Send + Sync { 
    /// Executes a text completion for the given prompt.
    async fn completion(&self, prompt: &str) -> anyhow::Result<String>; 
}

/// Standardized request for model providers.
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct Request { 
    /// Input prompt.
    pub prompt: String, 
    /// Current conversation context.
    pub context: Context, 
}

/// Standardized response for model providers.
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct Response { 
    /// Generated content text.
    pub content: String, 
    /// List of tool calls generated by the model.
    pub tool_calls: Vec<ToolCall>, 
}

/// A chunk of streaming content from a model.
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct ResponseChunk { 
    /// Partial content delta.
    pub content_delta: Option<String>, 
    /// Partial tool call delta.
    pub tool_call_delta: Option<ToolCallDelta>, 
    /// Indicates if the stream is finished.
    pub is_final: bool, 
}

/// Internal delta for tool call streaming.
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct ToolCallDelta { pub name: Option<String>, pub arguments_delta: Option<String>, }
/// A specific tool execution request from the AI.
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct ToolCall { pub name: String, pub arguments: serde_json::Value, }

/// A provider bridging an external AI model into the MindPalace ecosystem.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Non-streaming completion.
    async fn complete(&self, req: Request) -> anyhow::Result<Response>;
    /// Streaming completion returning a boxed stream of chunks.
    async fn stream_complete(&self, req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>>;
}

/// Local LLM provider via Ollama.
pub struct OllamaProvider { pub client: ollama_rs::Ollama, pub model: String, pub embedding_model: String, }
impl OllamaProvider { pub fn new(model: String, embedding_model: String) -> Self { Self { client: ollama_rs::Ollama::default(), model, embedding_model } } }
#[async_trait] impl LlmClient for OllamaProvider { async fn completion(&self, prompt: &str) -> anyhow::Result<String> { use ollama_rs::generation::completion::request::GenerationRequest; let res = self.client.generate(GenerationRequest::new(self.model.clone(), prompt.to_string())).await?; Ok(res.response) } }
#[async_trait] impl ModelProvider for OllamaProvider { async fn complete(&self, req: Request) -> anyhow::Result<Response> { let content = self.completion(&req.prompt).await?; Ok(Response { content, tool_calls: vec![] }) } async fn stream_complete(&self, req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> { use ollama_rs::generation::completion::request::GenerationRequest; let mut stream = self.client.generate_stream(GenerationRequest::new(self.model.clone(), req.prompt)).await?; let stream = async_stream::try_stream! { while let Some(res) = stream.next().await { let res_vec = res.map_err(|e| anyhow::anyhow!(e))?; for res_item in res_vec { yield ResponseChunk { content_delta: Some(res_item.response), tool_call_delta: None, is_final: res_item.done }; if res_item.done { break; } } } }; Ok(Box::pin(stream)) } }

/// Anthropic API provider.
pub struct AnthropicProvider { pub api_key: String, pub model: String, }
impl AnthropicProvider { pub fn new(api_key: String, model: String) -> Self { Self { api_key, model } } }
#[async_trait] impl LlmClient for AnthropicProvider { async fn completion(&self, _prompt: &str) -> anyhow::Result<String> { Ok("Anthropic completion stub".to_string()) } }
#[async_trait] impl ModelProvider for AnthropicProvider { async fn complete(&self, _req: Request) -> anyhow::Result<Response> { Ok(Response { content: "Anthropic response".into(), tool_calls: vec![] }) } async fn stream_complete(&self, _req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> { let stream = async_stream::try_stream! { yield ResponseChunk { content_delta: Some("Anthropic ".into()), tool_call_delta: None, is_final: false }; yield ResponseChunk { content_delta: Some("stream stub".into()), tool_call_delta: None, is_final: true }; }; Ok(Box::pin(stream)) } }

/// OpenAI API provider.
pub struct OpenAiProvider { pub api_key: String, pub model: String, }
impl OpenAiProvider { pub fn new(api_key: String, model: String) -> Self { Self { api_key, model } } }
#[async_trait] impl LlmClient for OpenAiProvider { async fn completion(&self, _prompt: &str) -> anyhow::Result<String> { Ok("OpenAI completion stub".to_string()) } }
#[async_trait] impl ModelProvider for OpenAiProvider { async fn complete(&self, _req: Request) -> anyhow::Result<Response> { Ok(Response { content: "OpenAI response".into(), tool_calls: vec![] }) } async fn stream_complete(&self, _req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> { let stream = async_stream::try_stream! { yield ResponseChunk { content_delta: Some("OpenAI ".into()), tool_call_delta: None, is_final: false }; yield ResponseChunk { content_delta: Some("stream stub".into()), tool_call_delta: None, is_final: true }; }; Ok(Box::pin(stream)) } }

/// Google Gemini API provider.
pub struct GeminiProvider { pub api_key: String, pub model: String, }
impl GeminiProvider { pub fn new(api_key: String, model: String) -> Self { Self { api_key, model } } }
#[async_trait] impl LlmClient for GeminiProvider { async fn completion(&self, _prompt: &str) -> anyhow::Result<String> { Ok("Gemini completion stub".to_string()) } }
#[async_trait] impl ModelProvider for GeminiProvider { async fn complete(&self, _req: Request) -> anyhow::Result<Response> { Ok(Response { content: "Gemini response".into(), tool_calls: vec![] }) } async fn stream_complete(&self, _req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> { let stream = async_stream::try_stream! { yield ResponseChunk { content_delta: Some("Gemini ".into()), tool_call_delta: None, is_final: false }; yield ResponseChunk { content_delta: Some("stream stub".into()), tool_call_delta: None, is_final: true }; }; Ok(Box::pin(stream)) } }

impl TokenCounter for GeminiProvider { fn count_tokens(&self, text: &str) -> usize { tiktoken_rs::cl100k_base().unwrap().encode_with_special_tokens(text).len() } }
impl TokenCounter for OllamaProvider { fn count_tokens(&self, text: &str) -> usize { tiktoken_rs::cl100k_base().unwrap().encode_with_special_tokens(text).len() } }
impl TokenCounter for AnthropicProvider { fn count_tokens(&self, text: &str) -> usize { tiktoken_rs::cl100k_base().unwrap().encode_with_special_tokens(text).len() } }
impl TokenCounter for OpenAiProvider { fn count_tokens(&self, text: &str) -> usize { let bpe = tiktoken_rs::get_bpe_from_model(&self.model).unwrap_or_else(|_| tiktoken_rs::cl100k_base().unwrap()); bpe.encode_with_special_tokens(text).len() } }

/// Scoring interface for determining how relevant a memory item is to the current conversation.
#[async_trait]
pub trait RelevanceAnalyzer: Send + Sync {
    /// Returns a score (0.0 to 1.0) indicating relevance level.
    async fn score_relevance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32>;
}

/// Scoring interface for determining the objective importance of a memory item for long-term retention.
#[async_trait]
pub trait ImportanceAnalyzer: Send + Sync {
    /// Returns a score (0.0 to 1.0) indicating base importance.
    async fn score_importance(&self, item: &MemoryItem, context: &Context) -> anyhow::Result<f32>;
}

/// Collection of Prometheus metrics for monitoring memory system health.
#[derive(Clone)] pub struct MemoryMetrics { 
    /// Total serialized size of current context in bytes.
    pub context_size_bytes: Gauge, 
    /// Total number of messages in context.
    pub item_count: Gauge, 
    /// Current context compression ratio.
    pub compression_ratio: Gauge, 
    /// Histogram of processing latencies for different memory layers.
    pub layer_latency: Histogram, 
    /// Total number of active facts in the graph.
    pub fact_count: Gauge, 
    /// Total tokens processed across the system.
    pub total_tokens_processed: Gauge, 
}
impl MemoryMetrics { pub fn new(registry: &Registry) -> anyhow::Result<Self> { let context_size_bytes = register_gauge_with_registry!(opts!("mindpalace_context_size_bytes", "desc"), registry)?; let item_count = register_gauge_with_registry!(opts!("mindpalace_item_count", "desc"), registry)?; let compression_ratio = register_gauge_with_registry!(opts!("mindpalace_compression_ratio", "desc"), registry)?; let layer_latency = register_histogram_with_registry!("mindpalace_layer_latency_seconds", "desc", vec![0.1], registry)?; let fact_count = register_gauge_with_registry!(opts!("mindpalace_fact_count", "desc"), registry)?; let total_tokens_processed = register_gauge_with_registry!(opts!("mindpalace_total_tokens_processed", "desc"), registry)?; Ok(Self { context_size_bytes, item_count, compression_ratio, layer_latency, fact_count, total_tokens_processed }) } }

pub mod analysis;

/// Utility functions for memory compression and comparison.
pub mod utils { 
    /// Estimates token count based on typical character density.
    pub fn estimate_tokens(text: &str) -> usize { text.len() / 4 } 
    /// Compresses binary data using Zstandard.
    pub fn compress(data: &[u8]) -> anyhow::Result<Vec<u8>> { Ok(zstd::encode_all(data, 3)?) } 
    /// Decompresses Zstandard binary data.
    pub fn decompress(data: &[u8]) -> anyhow::Result<Vec<u8>> { Ok(zstd::decode_all(data)?) }
    /// Calculates the cosine similarity between two float vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x,y)| x*y).sum();
        let n_a: f32 = a.iter().map(|x| x*x).sum::<f32>().sqrt();
        let n_b: f32 = b.iter().map(|x| x*x).sum::<f32>().sqrt();
        if n_a == 0.0 || n_b == 0.0 { 0.0 } else { dot / (n_a * n_b) }
    }
}
