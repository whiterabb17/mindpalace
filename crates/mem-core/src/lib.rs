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

/// Determines the visibility and sharing policy of a piece of knowledge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FactScope {
    /// Agent-specific or session-specific facts (default).
    Private,
    /// Shared across agents within the same project.
    Project,
    /// High-confidence technical or objective facts shared across the ecosystem.
    Global,
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
    /// Visibility scope for multi-agent learning.
    pub scope: FactScope,
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
            embedding: None,
            scope: FactScope::Private 
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
        props.insert("scope".into(), PropertyValue::from(match self.scope {
            FactScope::Private => "Private",
            FactScope::Project => "Project",
            FactScope::Global => "Global",
        }));
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
            scope: match get_str("scope").as_str() {
                "Project" => FactScope::Project,
                "Global" => FactScope::Global,
                _ => FactScope::Private,
            },
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

    /// Removes facts from the active set if they have passed their expiration date.
    pub fn garbage_collect_stale_facts(&self) -> anyhow::Result<usize> {
        let now = chrono::Utc::now().timestamp() as u64;
        let mut to_remove = Vec::new();
        
        {
            let ids = self.node_ids.read().unwrap();
            for id in ids.iter() {
                if let Some(fact) = self.get_fact(id) {
                    if let Some(expiry) = fact.valid_until {
                        if now > expiry {
                            to_remove.push(id.clone());
                        }
                    }
                }
            }
        }

        let count = to_remove.len();
        if count > 0 {
            let mut ids = self.node_ids.write().unwrap();
            for id in to_remove {
                ids.remove(&id);
                // Note: We don't delete from GraphDB here to preserve history/traces, 
                // but it's effectively removed from MindPalace's active context.
            }
        }
        
        Ok(count)
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

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce, Key
};

/// A decorator that provides transparent authenticated encryption for a StorageBackend.
///
/// Uses AES-256-GCM (AEAD) to ensure both confidentiality and integrity of session 
/// data at rest.
pub struct EncryptedStorageBackend<S: StorageBackend> {
    inner: S,
    cipher: Aes256Gcm,
}

impl<S: StorageBackend> EncryptedStorageBackend<S> {
    /// Initializes a new encryption layer with the provided 32-byte key.
    pub fn new(inner: S, key_bytes: [u8; 32]) -> Self {
        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);
        Self { inner, cipher }
    }
}

#[async_trait]
impl<S: StorageBackend> StorageBackend for EncryptedStorageBackend<S> {
    async fn store(&self, id: &str, data: &[u8]) -> anyhow::Result<()> {
        use rand::RngCore;
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt the data.
        let ciphertext = self.cipher.encrypt(nonce, data)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {:?}", e))?;
        
        // Prepend nonce to the stored payload.
        let mut payload = nonce_bytes.to_vec();
        payload.extend(ciphertext);
        
        self.inner.store(id, &payload).await
    }

    async fn retrieve(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        let payload = self.inner.retrieve(id).await?;
        if payload.len() < 12 { return Err(anyhow::anyhow!("Payload too short for decryption (missing nonce)")); }
        
        let (nonce_bytes, ciphertext) = payload.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        let plaintext = self.cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("Decryption failed: {:?}", e))?;
            
        Ok(plaintext)
    }

    async fn exists(&self, id: &str) -> bool { self.inner.exists(id).await }

    async fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>> { self.inner.list(prefix).await }
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
    pub context: Arc<Context>, 
    /// Available tool definitions for discovery.
    pub tools: Vec<ToolDefinition>,
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
    /// Precise token usage reported by the provider (typically only in the final chunk).
    pub usage: Option<ResponseUsage>,
    /// Indicates if the stream is finished.
    pub is_final: bool, 
}

/// Precise token metrics reported by an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct ResponseUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Internal delta for tool call streaming.
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct ToolCallDelta { pub name: Option<String>, pub arguments_delta: Option<String>, }
/// Metadata for discovering available tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

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
#[async_trait]
impl ModelProvider for OllamaProvider {
    async fn complete(&self, req: Request) -> anyhow::Result<Response> {
        let content = self.completion(&req.prompt).await?;
        Ok(Response { content, tool_calls: vec![] })
    }
    async fn stream_complete(&self, req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        use ollama_rs::generation::completion::request::GenerationRequest;
        let mut stream = self.client.generate_stream(GenerationRequest::new(self.model.clone(), req.prompt.clone())).await?;
        let stream = async_stream::try_stream! {
            while let Some(res) = stream.next().await {
                let res_vec = res.map_err(|e| anyhow::anyhow!(e))?;
                for res_item in res_vec {
                    yield ResponseChunk { content_delta: Some(res_item.response), tool_call_delta: None, usage: None, is_final: res_item.done };
                    if res_item.done { break; }
                }
            }
        };
        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
        let res = self.client.generate_embeddings(GenerateEmbeddingsRequest::new(
            self.embedding_model.clone(), 
            text.to_string().into()
        )).await?;
        Ok(res.embeddings.into_iter().next().unwrap_or_default())
    }
}

/// Anthropic API provider.
pub struct AnthropicProvider { pub api_key: String, pub model: String, }
impl AnthropicProvider { pub fn new(api_key: String, model: String) -> Self { Self { api_key, model } } }

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart {},
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: AnthropicDelta },
    #[serde(rename = "message_delta")]
    MessageDelta { usage: Option<AnthropicUsage> },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct AnthropicDelta {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[async_trait]
impl LlmClient for AnthropicProvider {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String> {
        let client = reqwest::Client::new();
        let messages = vec![AnthropicMessage { role: "user".into(), content: prompt.into() }];
        let req = AnthropicRequest { model: self.model.clone(), max_tokens: 4096, messages, stream: false };
        
        let res = client.post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&req)
            .send().await?;
        
        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("Anthropic API error: {}", err);
        }
        
        let data: AnthropicResponse = res.json().await?;
        Ok(data.content.first().map(|c| c.text.clone()).unwrap_or_default())
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    async fn complete(&self, req: Request) -> anyhow::Result<Response> {
        let content = self.completion(&req.prompt).await?;
        Ok(Response { content, tool_calls: vec![] })
    }

    async fn stream_complete(&self, req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let client = reqwest::Client::new();
        let messages = vec![AnthropicMessage { role: "user".into(), content: req.prompt.clone() }];
        let anthropic_req = AnthropicRequest { model: self.model.clone(), max_tokens: 4096, messages, stream: true };

        let res = client.post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_req)
            .send().await?;

        let stream = async_stream::try_stream! {
            let mut byte_stream = res.bytes_stream();
            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk?;
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    if line.starts_with("data: ") {
                        let data = line.trim_start_matches("data: ");
                        if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                            match event {
                                AnthropicStreamEvent::ContentBlockDelta { delta } => {
                                    yield ResponseChunk { content_delta: Some(delta.text), tool_call_delta: None, usage: None, is_final: false };
                                },
                                AnthropicStreamEvent::MessageDelta { usage } => {
                                    let response_usage = usage.map(|u| ResponseUsage {
                                        prompt_tokens: u.input_tokens,
                                        completion_tokens: u.output_tokens,
                                        total_tokens: u.input_tokens + u.output_tokens,
                                    });
                                    yield ResponseChunk { content_delta: None, tool_call_delta: None, usage: response_usage, is_final: true };
                                },
                                _ => {}
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl EmbeddingProvider for AnthropicProvider {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        // Anthropic does not provide a native embedding API.
        // Return a zero vector (1536 dims) as a placeholder.
        Ok(vec![0.0; 1536])
    }
}

/// OpenAI API provider.
pub struct OpenAiProvider { pub api_key: String, pub model: String, }
impl OpenAiProvider { pub fn new(api_key: String, model: String) -> Self { Self { api_key, model } } }

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    stream: bool,
    stream_options: Option<OpenAiStreamOptions>,
}

#[derive(Serialize)]
struct OpenAiStreamOptions {
    include_usage: bool,
}

#[derive(Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct OpenAiStreamResponse {
    choices: Vec<OpenAiStreamChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
}

#[async_trait]
impl LlmClient for OpenAiProvider {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String> {
        let client = reqwest::Client::new();
        let messages = vec![OpenAiMessage { role: "user".into(), content: prompt.into() }];
        let req = OpenAiRequest { model: self.model.clone(), messages, stream: false, stream_options: None };
        
        let res = client.post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&req)
            .send().await?;
            
        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("OpenAI API error: {}", err);
        }
            
        let data: OpenAiResponse = res.json().await?;
        Ok(data.choices.first().map(|c| c.message.content.clone()).unwrap_or_default())
    }
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    async fn complete(&self, req: Request) -> anyhow::Result<Response> {
        let content = self.completion(&req.prompt).await?;
        Ok(Response { content, tool_calls: vec![] })
    }

    async fn stream_complete(&self, req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let client = reqwest::Client::new();
        let messages = vec![OpenAiMessage { role: "user".into(), content: req.prompt.clone() }];
        let openai_req = OpenAiRequest { 
            model: self.model.clone(), 
            messages, 
            stream: true, 
            stream_options: Some(OpenAiStreamOptions { include_usage: true }) 
        };

        let res = client.post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&openai_req)
            .send().await?;

        let stream = async_stream::try_stream! {
            let mut byte_stream = res.bytes_stream();
            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk?;
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    if line.starts_with("data: ") {
                        let data = line.trim_start_matches("data: ");
                        if data == "[DONE]" { break; }
                        if let Ok(event) = serde_json::from_str::<OpenAiStreamResponse>(data) {
                            let usage = event.usage.map(|u| ResponseUsage {
                                prompt_tokens: u.prompt_tokens,
                                completion_tokens: u.completion_tokens,
                                total_tokens: u.total_tokens,
                            });

                            if let Some(choice) = event.choices.first() {
                                let is_final = choice.finish_reason.is_some();
                                yield ResponseChunk { content_delta: choice.delta.content.clone(), tool_call_delta: None, usage, is_final };
                            } else if usage.is_some() {
                                // OpenAI usage-only chunk
                                yield ResponseChunk { content_delta: None, tool_call_delta: None, usage, is_final: true };
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiProvider {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        #[derive(Serialize)]
        struct EmbRequest { input: String, model: String }
        #[derive(Deserialize)]
        struct EmbResponse { data: Vec<EmbData> }
        #[derive(Deserialize)]
        struct EmbData { embedding: Vec<f32> }

        let client = reqwest::Client::new();
        let model = if self.model.contains("gpt") { "text-embedding-3-small" } else { &self.model };
        let res = client.post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&EmbRequest { input: text.to_string(), model: model.to_string() })
            .send().await?;
        
        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("OpenAI Embedding error: {}", err);
        }
        
        let data: EmbResponse = res.json().await?;
        Ok(data.data.first().map(|d| d.embedding.clone()).unwrap_or_default())
    }
}

/// Google Gemini API provider.
pub struct GeminiProvider { pub api_key: String, pub model: String, }
impl GeminiProvider { pub fn new(api_key: String, model: String) -> Self { Self { api_key, model } } }

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[async_trait]
impl LlmClient for GeminiProvider {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String> {
        let client = reqwest::Client::new();
        let contents = vec![GeminiContent { role: "user".into(), parts: vec![GeminiPart { text: prompt.into() }] }];
        let req = GeminiRequest { contents };
        
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", self.model, self.api_key);
        let res = client.post(&url)
            .json(&req)
            .send().await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("Gemini API error: {}", err);
        }

        let data: GeminiResponse = res.json().await?;
        Ok(data.candidates.first().and_then(|c| c.content.parts.first()).map(|p| p.text.clone()).unwrap_or_default())
    }
}

#[async_trait]
impl ModelProvider for GeminiProvider {
    async fn complete(&self, req: Request) -> anyhow::Result<Response> {
        let content = self.completion(&req.prompt).await?;
        Ok(Response { content, tool_calls: vec![] })
    }

    async fn stream_complete(&self, req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let client = reqwest::Client::new();
        let contents = vec![GeminiContent { role: "user".into(), parts: vec![GeminiPart { text: req.prompt }] }];
        let gemini_req = GeminiRequest { contents };
        
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}", self.model, self.api_key);
        let res = client.post(&url)
            .json(&gemini_req)
            .send().await?;

        let stream = async_stream::try_stream! {
            let mut byte_stream = res.bytes_stream();
            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk?;
                if let Ok(event) = serde_json::from_slice::<Vec<GeminiResponse>>(&bytes) {
                    for res in event {
                        if let Some(candidate) = res.candidates.first() {
                            if let Some(part) = candidate.content.parts.first() {
                                yield ResponseChunk { content_delta: Some(part.text.clone()), tool_call_delta: None, usage: None, is_final: false };
                            }
                        }
                    }
                }
            }
            yield ResponseChunk { content_delta: None, tool_call_delta: None, usage: None, is_final: true };
        };

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl EmbeddingProvider for GeminiProvider {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        #[derive(Serialize)]
        struct GemEmbRequest { content: GemEmbContent }
        #[derive(Serialize)]
        struct GemEmbContent { parts: Vec<GemEmbPart> }
        #[derive(Serialize)]
        struct GemEmbPart { text: String }
        #[derive(Deserialize)]
        struct GemEmbResponse { embedding: GemEmbValue }
        #[derive(Deserialize)]
        struct GemEmbValue { values: Vec<f32> }

        let client = reqwest::Client::new();
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={}", self.api_key);
        let req = GemEmbRequest { content: GemEmbContent { parts: vec![GemEmbPart { text: text.to_string() }] } };
        let res = client.post(&url).json(&req).send().await?;
        
        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("Gemini Embedding error: {}", err);
        }
        
        let data: GemEmbResponse = res.json().await?;
        Ok(data.embedding.values)
    }
}

impl TokenCounter for GeminiProvider { fn count_tokens(&self, text: &str) -> usize { (text.len() / 4).max(1) } }
impl TokenCounter for OllamaProvider { fn count_tokens(&self, text: &str) -> usize { tiktoken_rs::cl100k_base().unwrap().encode_with_special_tokens(text).len() } }
impl TokenCounter for AnthropicProvider { fn count_tokens(&self, text: &str) -> usize { (text.len() / 3).max(1) } }
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

pub mod config;
pub use config::MindPalaceConfig;

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
