use async_trait::async_trait;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use prometheus::{
    opts, register_gauge_with_registry, register_histogram_with_registry, Gauge, Histogram,
    Registry,
};
use ruvector_graph::{Edge, GraphDB, Label, Node, Properties, PropertyValue};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use once_cell::sync::Lazy;
use tiktoken_rs::cl100k_base;

static TOKENIZER: Lazy<tiktoken_rs::CoreBPE> = Lazy::new(|| {
    cl100k_base().expect("Failed to initialize tiktoken")
});

/// Global token estimator - uses tiktoken-rs with a stable singleton pattern.
pub fn estimate_tokens(text: &str) -> usize {
    TOKENIZER.encode_with_special_tokens(text).len()
}

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
    System,
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
    pub fn new(
        content: String,
        category: String,
        confidence: f32,
        source_session_id: String,
    ) -> Self {
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
            scope: FactScope::Private,
        }
    }

    /// Converts the node into property-value pairs for GraphDB storage.
    pub fn to_properties(&self) -> Properties {
        let mut props = Properties::new();
        props.insert("content".into(), PropertyValue::from(self.content.clone()));
        props.insert(
            "category".into(),
            PropertyValue::from(self.category.clone()),
        );
        props.insert(
            "confidence".into(),
            PropertyValue::from(self.confidence as f64),
        );
        props.insert(
            "timestamp".into(),
            PropertyValue::from(self.timestamp as i64),
        );
        props.insert("version".into(), PropertyValue::from(self.version as i32));
        props.insert(
            "source_session_id".into(),
            PropertyValue::from(self.source_session_id.clone()),
        );
        props.insert(
            "scope".into(),
            PropertyValue::from(match self.scope {
                FactScope::Private => "Private",
                FactScope::Project => "Project",
                FactScope::Global => "Global",
            }),
        );
        if let Some(ref emb) = self.embedding {
            props.insert("embedding".into(), PropertyValue::from(emb.clone()));
        }
        props
    }

    /// Reconstructs a FactNode from GraphDB properties.
    pub fn from_properties(id: String, props: &Properties) -> Self {
        let get_str = |key: &str| {
            props
                .get(key)
                .and_then(|v| serde_json::to_value(v).ok())
                .and_then(|jv| {
                    jv.as_str()
                        .map(|s| s.to_string())
                        .or_else(|| jv.get("String").and_then(|s| s.as_str()).map(|s| s.to_string()))
                })
                .unwrap_or_default()
        };
        let get_f32 = |key: &str| {
            props
                .get(key)
                .and_then(|v| serde_json::to_value(v).ok())
                .and_then(|jv| {
                    jv.as_f64()
                        .map(|f| f as f32)
                        .or_else(|| jv.get("Float").and_then(|f| f.as_f64()).map(|f| f as f32))
                })
                .unwrap_or(0.0)
        };
        let get_u64 = |key: &str| {
            props
                .get(key)
                .and_then(|v| serde_json::to_value(v).ok())
                .and_then(|jv| {
                    jv.as_u64()
                        .or_else(|| {
                            jv.get("Int")
                                .and_then(|i| i.as_i64())
                                .map(|u| u as u64)
                        })
                        .or_else(|| {
                            jv.get("Integer")
                                .and_then(|i| i.as_i64())
                                .map(|u| u as u64)
                        })
                })
                .unwrap_or(0)
        };
        let get_u32 = |key: &str| {
            props
                .get(key)
                .and_then(|v| serde_json::to_value(v).ok())
                .and_then(|jv| {
                    jv.as_u64()
                        .map(|u| u as u32)
                        .or_else(|| {
                            jv.get("Int")
                                .and_then(|i| i.as_i64())
                                .map(|u| u as u32)
                        })
                        .or_else(|| {
                            jv.get("Integer")
                                .and_then(|i| i.as_i64())
                                .map(|u| u as u32)
                        })
                })
                .unwrap_or(1)
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
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let ids = self.node_ids.read().unwrap();
        let facts: Vec<FactNode> = ids.iter().filter_map(|id| self.get_fact(id)).collect();
        facts.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FactGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let facts = Vec::<FactNode>::deserialize(deserializer)?;
        let graph = FactGraph::new(None).map_err(serde::de::Error::custom)?;
        for fact in facts {
            let _ = graph.add_fact(fact);
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
        self.db
            .get_node(id)
            .map(|node| FactNode::from_properties(node.id.clone(), &node.properties))
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
        let node = Node::new(
            fact.id.clone(),
            vec![Label::new("Fact")],
            fact.to_properties(),
        );
        {
            let mut ids = self.node_ids.write().unwrap();
            ids.insert(fact.id.clone());
        }
        self.db
            .create_node(node)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok(())
    }

    /// Creates a directional link marking an old fact as superseded by a newer one.
    pub fn link_superseded(&self, old_id: &str, new_id: &str) -> anyhow::Result<()> {
        let edge = Edge::new(
            Uuid::new_v4().to_string(),
            old_id.to_string(),
            new_id.to_string(),
            "SUPERSEDED_BY".into(),
            Properties::new(),
        );
        self.db
            .create_edge(edge)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok(())
    }

    /// Creates a directional link indicating that one fact relies on another.
    pub fn link_dependency(&self, source_id: &str, target_id: &str) -> anyhow::Result<()> {
        let edge = Edge::new(
            Uuid::new_v4().to_string(),
            source_id.to_string(),
            target_id.to_string(),
            "DEPENDS_ON".into(),
            Properties::new(),
        );
        self.db
            .create_edge(edge)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
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
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}

/// The top-level knowledge container aggregating the fact graph and metadata.
#[derive(Serialize, Deserialize)]
pub struct KnowledgeBase {
    /// The relational fact graph.
    pub graph: FactGraph,
}

impl KnowledgeBase {
    /// Initializes a new KnowledgeBase at the specified location.
    pub fn new(path: Option<PathBuf>) -> anyhow::Result<Self> {
        Ok(Self {
            graph: FactGraph::new(path)?,
        })
    }
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
#[derive(Clone)]
pub struct FileStorage {
    /// The root path for data storage.
    pub root: PathBuf,
}
impl FileStorage {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}
#[async_trait]
impl StorageBackend for FileStorage {
    async fn store(&self, id: &str, data: &[u8]) -> anyhow::Result<()> {
        let path = self.root.join(id);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, data).await?;
        Ok(())
    }
    async fn retrieve(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        let path = self.root.join(id);
        Ok(tokio::fs::read(path).await?)
    }
    async fn exists(&self, id: &str) -> bool {
        self.root.join(id).exists()
    }
    async fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>> {
        let scan_path = self.root.join(prefix);
        if !scan_path.exists() {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(scan_path).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                entries.push(name.to_string());
            }
        }
        Ok(entries)
    }
}

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
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
        let ciphertext = self
            .cipher
            .encrypt(nonce, data)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {:?}", e))?;

        // Prepend nonce to the stored payload.
        let mut payload = nonce_bytes.to_vec();
        payload.extend(ciphertext);

        self.inner.store(id, &payload).await
    }

    async fn retrieve(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        let payload = self.inner.retrieve(id).await?;
        if payload.len() < 12 {
            return Err(anyhow::anyhow!(
                "Payload too short for decryption (missing nonce)"
            ));
        }

        let (nonce_bytes, ciphertext) = payload.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let plaintext = self
            .cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("Decryption failed: {:?}", e))?;

        Ok(plaintext)
    }

    async fn exists(&self, id: &str) -> bool {
        self.inner.exists(id).await
    }

    async fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>> {
        self.inner.list(prefix).await
    }
}

/// High-level client for executing model completions.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Executes a text completion for the given prompt.
    async fn completion(&self, prompt: &str) -> anyhow::Result<String>;
}

/// Standardized request for model providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Input prompt.
    pub prompt: String,
    /// Current conversation context.
    pub context: Arc<Context>,
    /// Available tool definitions for discovery.
    pub tools: Vec<ToolDefinition>,
}

/// Standardized response for model providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Generated content text.
    pub content: String,
    /// List of tool calls generated by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage metrics for the response.
    pub usage: Option<ResponseUsage>,
}

impl Response {
    pub fn new(content: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            content,
            tool_calls,
            usage: None,
        }
    }
}

/// A chunk of streaming content from a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChunk {
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Internal delta for tool call streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub name: Option<String>,
    pub arguments_delta: Option<String>,
    pub id: Option<String>,
}
/// Metadata for discovering available tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A specific tool execution request from the AI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub id: String,
}

pub struct ModelMetadata {
    pub name: String,
    pub context_window: usize,
}

/// A provider bridging an external AI model into the MindPalace ecosystem.
#[async_trait]
pub trait ModelProvider: LlmClient + Send + Sync {
    /// Discovers metadata about the model (e.g. context window) from the provider API.
    async fn discover_metadata(&self) -> ModelMetadata {
        ModelMetadata {
            name: "unknown".to_string(),
            context_window: 2048,
        }
    }
    /// Dynamically updates the context window size for subsequent calls.
    fn set_context_window(&self, _size: usize) {}
    /// Non-streaming completion.
    async fn complete(&self, req: Request) -> anyhow::Result<Response>;
    /// Streaming completion returning a boxed stream of chunks.
    async fn stream_complete(
        &self,
        req: Request,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>>;
}

/// Local LLM provider via Ollama.
pub struct OllamaProvider {
    pub client: ollama_rs::Ollama,
    pub model: String,
    pub embedding_model: String,
    pub num_ctx: std::sync::atomic::AtomicU32,
}

impl OllamaProvider {
    pub fn new(
        base_url: String,
        model: String,
        embedding_model: String,
        num_ctx: Option<u32>,
    ) -> Self {
        let url = reqwest::Url::parse(&base_url)
            .unwrap_or_else(|_| reqwest::Url::parse("http://127.0.0.1:11434").unwrap());
        let scheme = url.scheme();
        let mut host_val = url.host_str().unwrap_or("127.0.0.1").to_string();
        if host_val == "localhost" {
            host_val = "127.0.0.1".to_string();
        }
        let base = format!("{}://{}", scheme, host_val);
        let port = url.port().unwrap_or(11434);

        Self {
            client: ollama_rs::Ollama::new(base, port),
            model,
            embedding_model,
            num_ctx: std::sync::atomic::AtomicU32::new(num_ctx.unwrap_or(0)),
        }
    }

    /// Attempts to discover model metadata (like context window) directly from Ollama.
    pub async fn get_model_info(&self) -> anyhow::Result<ollama_rs::models::LocalModel> {
        let models = self.client.list_local_models().await?;
        models
            .into_iter()
            .find(|m| m.name == self.model || m.name.starts_with(&self.model))
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in Ollama inventory", self.model))
    }
}

#[async_trait]
impl LlmClient for OllamaProvider {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String> {
        use ollama_rs::generation::completion::request::GenerationRequest;
        let mut gen_req = GenerationRequest::new(self.model.clone(), prompt.to_string());
        let ctx = self.num_ctx.load(std::sync::atomic::Ordering::SeqCst);
        if ctx > 0 {
            gen_req = gen_req.options(
                ollama_rs::generation::options::GenerationOptions::default().num_ctx(ctx.into()),
            );
        }
        let res = self.client.generate(gen_req).await?;
        Ok(res.response)
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    async fn discover_metadata(&self) -> ModelMetadata {
        let ctx = self.num_ctx.load(std::sync::atomic::Ordering::SeqCst);
        let context_window = if ctx > 0 {
            ctx as usize
        } else {
            // Attempt to query Ollama for the actual model context window
            match self.client.list_local_models().await {
                Ok(models) => {
                    models
                        .into_iter()
                        .find(|m| m.name == self.model || m.name.starts_with(&self.model))
                        .map(|m| {
                            if m.name.contains("qwen2.5-coder") {
                                32768
                            } else if m.name.contains("llama3.2") || m.name.contains("llama3.1") {
                                131072
                            } else if m.name.contains("mistral") {
                                32768
                            } else {
                                2048
                            }
                        })
                        .unwrap_or(2048)
                }
                Err(_) => 2048,
            }
        };

        ModelMetadata {
            name: self.model.clone(),
            context_window,
        }
    }

    fn set_context_window(&self, size: usize) {
        self.num_ctx
            .store(size as u32, std::sync::atomic::Ordering::SeqCst);
    }

    async fn complete(&self, req: Request) -> anyhow::Result<Response> {
        let mut messages = Vec::new();
        for item in req.context.items.iter() {
            let role = match item.role {
                MemoryRole::User => "user",
                MemoryRole::Assistant => "assistant",
                MemoryRole::System => "system",
                MemoryRole::Tool => "tool",
            };
            let mut msg = serde_json::json!({
                "role": role,
                "content": item.content,
            });

            if let Some(tc) = item.metadata.get("tool_calls") {
                msg["tool_calls"] = tc.clone();
            }
            if let Some(tcid) = item.metadata.get("tool_call_id") {
                msg["tool_call_id"] = tcid.clone();
            }
            messages.push(msg);
        }

        // --- SECONDARY SAFETY: Sanitize message sequence ---
        let mut sanitized = Vec::new();
        let mut can_accept_tool = false;
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role == "assistant" {
                can_accept_tool = msg.get("tool_calls").map(|tc| !tc.is_null()).unwrap_or(false);
            } else if role == "tool" {
                if !can_accept_tool {
                    tracing::warn!("Ollama Provider: Dropping orphaned 'tool' message.");
                    continue;
                }
            } else {
                can_accept_tool = false;
            }
            sanitized.push(msg);
        }
        let mut messages = sanitized;



        messages.push(serde_json::json!({
            "role": "user",
            "content": req.prompt,
        }));

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });

        if !req.tools.is_empty() {
            let tools: Vec<serde_json::Value> = req
                .tools
                .into_iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), serde_json::Value::Array(tools));
        }

        let ctx = self.num_ctx.load(std::sync::atomic::Ordering::SeqCst);
        if ctx > 0 {
            body.as_object_mut().unwrap().insert(
                "options".to_string(),
                serde_json::json!({
                    "num_ctx": ctx
                }),
            );
        }

        let url = format!("{}api/chat", self.client.url_str());
        let res = reqwest::Client::new().post(url).json(&body).send().await?;

        if !res.status().is_success() {
            let status = res.status();
            let err_text = res.text().await.unwrap_or_default();
            anyhow::bail!("Ollama error ({}): {}", status, err_text);
        }

        let res_json: serde_json::Value = res.json().await?;

        let prompt_tokens = res_json["prompt_eval_count"].as_u64().unwrap_or(0);
        let completion_tokens = res_json["eval_count"].as_u64().unwrap_or(0);
        let usage = Some(ResponseUsage {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: completion_tokens as u32,
            total_tokens: (prompt_tokens + completion_tokens) as u32,
        });

        let content = res_json["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let mut tool_calls = Vec::new();

        if let Some(calls) = res_json["message"]["tool_calls"].as_array() {
            for call in calls {
                if let (Some(name), Some(args)) = (
                    call["function"]["name"].as_str(),
                    call["function"]["arguments"].as_object(),
                ) {
                    tool_calls.push(ToolCall {
                        name: name.to_string(),
                        arguments: serde_json::Value::Object(args.clone()),
                        id: uuid::Uuid::new_v4().to_string(),
                    });
                }
            }
        }

        // FALLBACK: Parse from content if no native tool_calls found
        if tool_calls.is_empty() && !content.is_empty() {
            // 1. Try to find XML-like tags <tool_call>... </tool_call>
            if let Some(start) = content.find("<tool_call>") {
                if let Some(end) = content[start..].find("</tool_call>") {
                    let json_str = &content[start + 11..start + end];
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                        if let (Some(name), Some(args)) = (val["name"].as_str(), val["arguments"].as_object()) {
                            tool_calls.push(ToolCall {
                                name: name.to_string(),
                                arguments: serde_json::Value::Object(args.clone()),
                                id: uuid::Uuid::new_v4().to_string(),
                            });
                        }
                    }
                }
            }
            
            // 2. Try to find markdown JSON blocks
            if tool_calls.is_empty() {
                if let Some(start) = content.find("```json") {
                    let search_area = &content[start + 7..];
                    if let Some(end) = search_area.find("```") {
                        let json_str = &search_area[..end];
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str.trim()) {
                             if let (Some(name), Some(args)) = (val["name"].as_str(), val["arguments"].as_object()) {
                                tool_calls.push(ToolCall {
                                    name: name.to_string(),
                                    arguments: serde_json::Value::Object(args.clone()),
                                    id: uuid::Uuid::new_v4().to_string(),
                                });
                            }
                        }
                    }
                } else if let Some(start) = content.find("```") {
                    let search_area = &content[start + 3..];
                    if let Some(end) = search_area.find("```") {
                        let json_str = &search_area[..end];
                         if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str.trim()) {
                             if let (Some(name), Some(args)) = (val["name"].as_str(), val["arguments"].as_object()) {
                                tool_calls.push(ToolCall {
                                    name: name.to_string(),
                                    arguments: serde_json::Value::Object(args.clone()),
                                    id: uuid::Uuid::new_v4().to_string(),
                                });
                            }
                        }
                    }
                }
            }

            // 3. Try to parse the whole content as JSON (some models just output raw JSON)
            if tool_calls.is_empty() {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(content.trim()) {
                     if let (Some(name), Some(args)) = (val["name"].as_str(), val["arguments"].as_object()) {
                        tool_calls.push(ToolCall {
                            name: name.to_string(),
                            arguments: serde_json::Value::Object(args.clone()),
                            id: uuid::Uuid::new_v4().to_string(),
                        });
                    }
                }
            }
        }

        Ok(Response {
            content,
            tool_calls,
            usage,
        })
    }

    async fn stream_complete(
        &self,
        req: Request,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let mut messages = Vec::new();
        for item in req.context.items.iter() {
            let role = match item.role {
                MemoryRole::User => "user",
                MemoryRole::Assistant => "assistant",
                MemoryRole::System => "system",
                MemoryRole::Tool => "tool",
            };
            let mut msg = serde_json::json!({
                "role": role,
                "content": item.content,
            });

            if let Some(tc) = item.metadata.get("tool_calls") {
                msg["tool_calls"] = tc.clone();
            }
            if let Some(tcid) = item.metadata.get("tool_call_id") {
                msg["tool_call_id"] = tcid.clone();
            }
            messages.push(msg);
        }

        // --- SECONDARY SAFETY: Sanitize message sequence ---
        let mut sanitized = Vec::new();
        let mut can_accept_tool = false;
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role == "assistant" {
                can_accept_tool = msg.get("tool_calls").map(|tc| !tc.is_null()).unwrap_or(false);
            } else if role == "tool" {
                if !can_accept_tool {
                    tracing::warn!("Ollama Provider: Dropping orphaned 'tool' message (streaming).");
                    continue;
                }
            } else {
                can_accept_tool = false;
            }
            sanitized.push(msg);
        }
        let mut messages = sanitized;


        messages.push(serde_json::json!({
            "role": "user",
            "content": req.prompt,
        }));

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
        });

        if !req.tools.is_empty() {
            let tools: Vec<serde_json::Value> = req
                .tools
                .into_iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), serde_json::Value::Array(tools));
        }

        let ctx = self.num_ctx.load(std::sync::atomic::Ordering::SeqCst);
        if ctx > 0 {
            body.as_object_mut().unwrap().insert(
                "options".to_string(),
                serde_json::json!({
                    "num_ctx": ctx
                }),
            );
        }

        let url = format!("{}api/chat", self.client.url_str());
        let res = reqwest::Client::new().post(url).json(&body).send().await?;

        if !res.status().is_success() {
            let status = res.status();
            let err_text = res.text().await.unwrap_or_default();
            anyhow::bail!("Ollama error ({}): {}", status, err_text);
        }

        let stream = res.bytes_stream();

        let stream = async_stream::try_stream! {
            let mut it = stream;
            // Use a labeled loop so we can break out of both levels when done=true.
            'outer: while let Some(chunk_res) = it.next().await {
                let bytes = chunk_res.map_err(|e| anyhow::anyhow!("Stream error: {}", e))?;
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    if line.trim().is_empty() { continue; }
                    let chunk: serde_json::Value = serde_json::from_str(line)
                        .map_err(|e| anyhow::anyhow!("JSON error: {}", e))?;

                    let is_done = chunk["done"].as_bool().unwrap_or(false);
                    let content_delta = chunk["message"]["content"].as_str().map(|s| s.to_string());

                    let usage = if is_done {
                        let prompt_tokens = chunk["prompt_eval_count"].as_u64().unwrap_or(0) as u32;
                        let completion_tokens = chunk["eval_count"].as_u64().unwrap_or(0) as u32;
                        if prompt_tokens > 0 || completion_tokens > 0 {
                            Some(ResponseUsage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // When the model requests tool calls, emit one ResponseChunk per call so
                    // the agent loop can capture ALL of them, not just the first one.
                    // Ollama places all tool calls in a single JSON line (usually the done=true
                    // chunk), so we iterate here and set is_final only on the last entry.
                    if let Some(calls) = chunk["message"]["tool_calls"].as_array() {
                        if !calls.is_empty() {
                            let num_calls = calls.len();
                            for (i, call) in calls.iter().enumerate() {
                                let is_last = i == num_calls - 1;
                                yield ResponseChunk {
                                    // Attach content_delta (usually empty) only to the first chunk
                                    // to avoid duplicating any preamble text.
                                    content_delta: if i == 0 { content_delta.clone() } else { None },
                                    tool_call_delta: Some(ToolCallDelta {
                                        name: call["function"]["name"].as_str().map(|s| s.to_string()),
                                        arguments_delta: Some(call["function"]["arguments"].to_string()),
                                        id: call["id"].as_str().map(|s| s.to_string()),
                                    }),
                                    usage: if is_last { usage.clone() } else { None },
                                    is_final: is_done && is_last,
                                };
                            }
                            if is_done { break 'outer; }
                            continue; // Skip the normal content yield below
                        }
                    }

                    // Normal path: content-only streaming chunk.
                    yield ResponseChunk {
                        content_delta,
                        tool_call_delta: None,
                        usage,
                        is_final: is_done,
                    };
                    if is_done { break 'outer; }
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
        let res = self
            .client
            .generate_embeddings(GenerateEmbeddingsRequest::new(
                self.embedding_model.clone(),
                text.to_string().into(),
            ))
            .await?;
        Ok(res.embeddings.into_iter().next().unwrap_or_default())
    }
}

/// Anthropic API provider.
pub struct AnthropicProvider {
    pub api_key: String,
    pub model: String,
}
impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}



#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContentUnion,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicContentUnion {
    Single(String),
    Multiple(Vec<AnthropicContent>),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}


#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart {},
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        _index: usize,
        content_block: AnthropicContent,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        _index: usize,
        delta: AnthropicDelta,
    },
    #[serde(rename = "message_delta")]
    MessageDelta { usage: Option<AnthropicUsage> },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
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
        let messages = vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContentUnion::Single(prompt.into()),
        }];
        let req = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            messages,
            tools: vec![],
            stream: false,
            system: None,
        };


        let res = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&req)
            .send()
            .await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("Anthropic API error: {}", err);
        }

        let data: AnthropicResponse = res.json().await?;
        let mut text = String::new();
        for item in data.content {
            if let AnthropicContent::Text { text: t } = item {
                text.push_str(&t);
            }
        }
        Ok(text)
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    async fn complete(&self, req: Request) -> anyhow::Result<Response> {
        let client = reqwest::Client::new();
        let mut messages = Vec::new();
        let mut system_prompt = None;

        for item in req.context.items.iter() {
            match item.role {
                MemoryRole::System => {
                    system_prompt = Some(item.content.clone());
                }
                MemoryRole::User => {
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContentUnion::Single(item.content.clone()),
                    });
                }
                MemoryRole::Assistant => {
                    let tool_calls: Option<Vec<ToolCall>> = item.metadata.get("tool_calls")
                        .and_then(|v| serde_json::from_value(v.clone()).ok());

                    if let Some(tcs) = tool_calls {
                        let mut content_blocks = Vec::new();
                        if !item.content.is_empty() {
                            content_blocks.push(AnthropicContent::Text { text: item.content.clone() });
                        }
                        for tc in tcs {
                            content_blocks.push(AnthropicContent::ToolUse {
                                id: tc.id,
                                name: tc.name,
                                input: tc.arguments,
                            });
                        }
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContentUnion::Multiple(content_blocks),
                        });
                    } else {
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContentUnion::Single(item.content.clone()),
                        });
                    }
                }
                MemoryRole::Tool => {
                    let tool_call_id = item.metadata.get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContentUnion::Multiple(vec![AnthropicContent::ToolResult {
                            tool_use_id: tool_call_id,
                            content: item.content.clone(),
                        }]),
                    });
                }
            }
        }

        messages.push(AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicContentUnion::Single(req.prompt),
        });

        // --- SECONDARY SAFETY: Sanitize message sequence for Anthropic ---
        let mut sanitized = Vec::new();
        let mut can_accept_tool = false;
        
        // DEBUG: Log the sequence we received
        let roles: Vec<String> = messages.iter().map(|m| m.role.clone()).collect();
        tracing::debug!("Anthropic Provider: Incoming roles: {:?}", roles);

        for msg in messages {
            let is_tool_response = match &msg.content {
                AnthropicContentUnion::Multiple(blocks) => {
                    blocks.iter().any(|b| matches!(b, AnthropicContent::ToolResult { .. }))
                }
                _ => false,
            };

            let has_tool_use = match &msg.content {
                AnthropicContentUnion::Multiple(blocks) => {
                    blocks.iter().any(|b| matches!(b, AnthropicContent::ToolUse { .. }))
                }
                _ => false,
            };

            if has_tool_use {
                can_accept_tool = true;
            } else if is_tool_response {
                if !can_accept_tool {
                    tracing::warn!("Anthropic Provider: Dropping orphaned tool result to avoid API error.");
                    continue;
                }
            } else {
                can_accept_tool = false;
            }
            sanitized.push(msg);
        }
        let messages = sanitized;




        let tools = req
            .tools
            .into_iter()
            .map(|t| AnthropicTool {
                name: t.name,
                description: t.description,
                input_schema: t.parameters,
            })
            .collect();

        let anthropic_req = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            messages,
            tools,
            stream: false,
            system: system_prompt,
        };


        let res = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_req)
            .send()
            .await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("Anthropic API error: {}", err);
        }

        let data: AnthropicResponse = res.json().await?;
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for item in data.content {
            match item {
                AnthropicContent::Text { text } => content.push_str(&text),
                AnthropicContent::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        name,
                        arguments: input,
                        id,
                    });
                }
                AnthropicContent::ToolResult { .. } => {} // Should not be returned by model
            }
        }


                let usage = Some(ResponseUsage {
            prompt_tokens: data.usage.input_tokens,
            completion_tokens: data.usage.output_tokens,
            total_tokens: data.usage.input_tokens + data.usage.output_tokens,
        });

        Ok(Response {
            content,
            tool_calls,
            usage,
        })
    }

    async fn stream_complete(
        &self,
        req: Request,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let client = reqwest::Client::new();
        let mut messages = Vec::new();
        let mut system_prompt = None;

        for item in req.context.items.iter() {
            match item.role {
                MemoryRole::System => {
                    system_prompt = Some(item.content.clone());
                }
                MemoryRole::User => {
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContentUnion::Single(item.content.clone()),
                    });
                }
                MemoryRole::Assistant => {
                    let tool_calls: Option<Vec<ToolCall>> = item.metadata.get("tool_calls")
                        .and_then(|v| serde_json::from_value(v.clone()).ok());

                    if let Some(tcs) = tool_calls {
                        let mut content_blocks = Vec::new();
                        if !item.content.is_empty() {
                            content_blocks.push(AnthropicContent::Text { text: item.content.clone() });
                        }
                        for tc in tcs {
                            content_blocks.push(AnthropicContent::ToolUse {
                                id: tc.id,
                                name: tc.name,
                                input: tc.arguments,
                            });
                        }
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContentUnion::Multiple(content_blocks),
                        });
                    } else {
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContentUnion::Single(item.content.clone()),
                        });
                    }
                }
                MemoryRole::Tool => {
                    let tool_call_id = item.metadata.get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContentUnion::Multiple(vec![AnthropicContent::ToolResult {
                            tool_use_id: tool_call_id,
                            content: item.content.clone(),
                        }]),
                    });
                }
            }
        }

        messages.push(AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicContentUnion::Single(req.prompt),
        });

        // --- SECONDARY SAFETY: Sanitize message sequence for Anthropic ---
        let mut sanitized = Vec::new();
        let mut can_accept_tool = false;
        for msg in messages {
            let is_tool_response = match &msg.content {
                AnthropicContentUnion::Multiple(blocks) => {
                    blocks.iter().any(|b| matches!(b, AnthropicContent::ToolResult { .. }))
                }
                _ => false,
            };

            let has_tool_use = match &msg.content {
                AnthropicContentUnion::Multiple(blocks) => {
                    blocks.iter().any(|b| matches!(b, AnthropicContent::ToolUse { .. }))
                }
                _ => false,
            };

            if has_tool_use {
                can_accept_tool = true;
            } else if is_tool_response {
                if !can_accept_tool {
                    tracing::warn!("Anthropic Provider: Dropping orphaned tool result to avoid API error (streaming).");
                    continue;
                }
            } else {
                can_accept_tool = false;
            }
            sanitized.push(msg);
        }
        let messages = sanitized;



        let tools = req
            .tools
            .into_iter()
            .map(|t| AnthropicTool {
                name: t.name,
                description: t.description,
                input_schema: t.parameters,
            })
            .collect();

        let anthropic_req = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            messages,
            tools,
            stream: true,
            system: system_prompt,
        };


        let res = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_req)
            .send()
            .await?;

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
                                AnthropicStreamEvent::ContentBlockStart { content_block, .. } => {
                                    match content_block {
                                        AnthropicContent::ToolUse { id, name, .. } => {
                                            yield ResponseChunk {
                                                content_delta: None,
                                                tool_call_delta: Some(ToolCallDelta { name: Some(name), arguments_delta: None, id: Some(id) }),
                                                usage: None,
                                                is_final: false
                                            };
                                        },
                                        _ => {} // Ignore Text blocks at start (handled in Delta) and ToolResults (not generated by model)
                                    }
                                },

                                AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                                    match delta {
                                        AnthropicDelta::TextDelta { text } => {
                                            yield ResponseChunk { content_delta: Some(text), tool_call_delta: None, usage: None, is_final: false };
                                        },
                                        AnthropicDelta::InputJsonDelta { partial_json } => {
                                            yield ResponseChunk {
                                                content_delta: None,
                                                tool_call_delta: Some(ToolCallDelta { name: None, arguments_delta: Some(partial_json), id: None }),
                                                usage: None,
                                                is_final: false
                                            };
                                        }
                                    }
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
pub struct OpenAiProvider {
    pub api_key: String,
    pub model: String,
}
impl OpenAiProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }
}

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiTool>,
    stream: bool,
    stream_options: Option<OpenAiStreamOptions>,
}

#[derive(Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    r#type: String,
    function: OpenAiFunction,
}

#[derive(Serialize)]
struct OpenAiFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct OpenAiStreamOptions {
    include_usage: bool,
}

#[derive(Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    r#type: String,
    function: OpenAiFunctionCall,
}

#[derive(Serialize, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
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
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct OpenAiToolCallDelta {
    index: Option<u32>,
    id: Option<String>,
    #[serde(rename = "type")]
    r#type: Option<String>,
    function: Option<OpenAiFunctionCallDelta>,
}

#[derive(Deserialize)]
struct OpenAiFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[async_trait]
impl LlmClient for OpenAiProvider {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String> {
        let client = reqwest::Client::new();
        let messages = vec![OpenAiMessage {
            role: "user".into(),
            content: Some(prompt.into()),
            tool_calls: None,
            tool_call_id: None,
        }];
        let req = OpenAiRequest {
            model: self.model.clone(),
            messages,
            tools: vec![],
            stream: false,
            stream_options: None,
        };

        let res = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("OpenAI API error: {}", err);
        }

        let data: OpenAiResponse = res.json().await?;
        Ok(data
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default())
    }
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    async fn complete(&self, req: Request) -> anyhow::Result<Response> {
        let client = reqwest::Client::new();
        let mut messages = Vec::new();

        for item in req.context.items.iter() {
            let role = match item.role {
                MemoryRole::User => "user",
                MemoryRole::Assistant => "assistant",
                MemoryRole::Tool => "tool",
                MemoryRole::System => "system",
            };

            let tool_calls = item
                .metadata
                .get("tool_calls")
                .and_then(|v| serde_json::from_value(v.clone()).ok());
            let tool_call_id = item
                .metadata
                .get("tool_call_id")
                .and_then(|v| v.as_str().map(|s| s.to_string()));

            messages.push(OpenAiMessage {
                role: role.to_string(),
                content: Some(item.content.clone()),
                tool_calls,
                tool_call_id,
            });
        }

        // --- SECONDARY SAFETY: Sanitize message sequence for OpenAI ---
        // OpenAI requires that 'tool' messages follow an 'assistant' message with 'tool_calls'.
        // Multiple 'tool' messages can follow a single 'assistant' call.
        let mut sanitized = Vec::new();
        let mut can_accept_tool = false;
        
        // DEBUG: Log the sequence we received
        let roles: Vec<String> = messages.iter().map(|m| m.role.clone()).collect();
        tracing::debug!("OpenAI Provider: Incoming roles: {:?}", roles);

        for msg in messages {
            if msg.role == "assistant" {
                can_accept_tool = msg.tool_calls.as_ref().map(|tc| !tc.is_empty()).unwrap_or(false);
            } else if msg.role == "tool" {
                if !can_accept_tool {
                    tracing::warn!(
                        "OpenAI Provider: Dropping orphaned 'tool' message (ID: {:?}). Sequence integrity was broken upstream.",
                        msg.tool_call_id
                    );
                    continue;
                }
            } else {
                can_accept_tool = false;
            }
            sanitized.push(msg);
        }
        let mut messages = sanitized;




        messages.push(OpenAiMessage {
            role: "user".to_string(),
            content: Some(req.prompt),
            tool_calls: None,
            tool_call_id: None,
        });

        let tools = req
            .tools
            .into_iter()
            .map(|t| OpenAiTool {
                r#type: "function".to_string(),
                function: OpenAiFunction {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                },
            })
            .collect();

        let openai_req = OpenAiRequest {
            model: self.model.clone(),
            messages,
            tools,
            stream: false,
            stream_options: None,
        };

        let res = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&openai_req)
            .send()
            .await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("OpenAI API error: {}", err);
        }

        let data: OpenAiResponse = res.json().await?;
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(choice) = data.choices.first() {
            if let Some(ref text) = choice.message.content {
                content.push_str(text);
            }
            if let Some(ref tcs) = choice.message.tool_calls {
                for tc in tcs {
                    if let Ok(args) = serde_json::from_str(&tc.function.arguments) {
                        tool_calls.push(ToolCall {
                            name: tc.function.name.clone(),
                            arguments: args,
                            id: tc.id.clone(),
                        });
                    }
                }
            }
        }

        let usage = data.usage.map(|u| ResponseUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(Response {
            content,
            tool_calls,
            usage,
        })
    }

    async fn stream_complete(
        &self,
        req: Request,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let client = reqwest::Client::new();
        let mut messages = Vec::new();

        for item in req.context.items.iter() {
            let role = match item.role {
                MemoryRole::User => "user",
                MemoryRole::Assistant => "assistant",
                MemoryRole::Tool => "tool",
                MemoryRole::System => "system",
            };

            let tool_calls = item
                .metadata
                .get("tool_calls")
                .and_then(|v| serde_json::from_value(v.clone()).ok());
            let tool_call_id = item
                .metadata
                .get("tool_call_id")
                .and_then(|v| v.as_str().map(|s| s.to_string()));

            messages.push(OpenAiMessage {
                role: role.to_string(),
                content: Some(item.content.clone()),
                tool_calls,
                tool_call_id,
            });
        }

        // --- SECONDARY SAFETY: Sanitize message sequence for OpenAI ---
        let mut sanitized = Vec::new();
        let mut can_accept_tool = false;
        for msg in messages {
            if msg.role == "assistant" {
                can_accept_tool = msg.tool_calls.as_ref().map(|tc| !tc.is_empty()).unwrap_or(false);
            } else if msg.role == "tool" {
                if !can_accept_tool {
                    tracing::warn!("OpenAI Provider: Dropping orphaned 'tool' message to prevent API error (streaming).");
                    continue;
                }
            } else {
                can_accept_tool = false;
            }
            sanitized.push(msg);
        }
        let mut messages = sanitized;



        messages.push(OpenAiMessage {
            role: "user".to_string(),
            content: Some(req.prompt),
            tool_calls: None,
            tool_call_id: None,
        });

        let tools = req
            .tools
            .into_iter()
            .map(|t| OpenAiTool {
                r#type: "function".to_string(),
                function: OpenAiFunction {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                },
            })
            .collect();

        let openai_req = OpenAiRequest {
            model: self.model.clone(),
            messages,
            tools,
            stream: true,
            stream_options: Some(OpenAiStreamOptions {
                include_usage: true,
            }),
        };

        let res = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&openai_req)
            .send()
            .await?;

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

                                if let Some(ref content) = choice.delta.content {
                                    yield ResponseChunk {
                                        content_delta: Some(content.clone()),
                                        tool_call_delta: None,
                                        usage: None,
                                        is_final: false
                                    };
                                }

                                if let Some(ref tcs) = choice.delta.tool_calls {
                                    for tc in tcs {
                                        let tool_call_delta = Some(ToolCallDelta {
                                            name: tc.function.as_ref().and_then(|f| f.name.clone()),
                                            arguments_delta: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                                            id: tc.id.clone(),
                                        });
                                        yield ResponseChunk {
                                            content_delta: None,
                                            tool_call_delta,
                                            usage: None,
                                            is_final: false
                                        };
                                    }
                                }

                                if is_final {
                                    yield ResponseChunk { content_delta: None, tool_call_delta: None, usage: usage.clone(), is_final: true };
                                }
                            } else if usage.is_some() {
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
        struct EmbRequest {
            input: String,
            model: String,
        }
        #[derive(Deserialize)]
        struct EmbResponse {
            data: Vec<EmbData>,
        }
        #[derive(Deserialize)]
        struct EmbData {
            embedding: Vec<f32>,
        }

        let client = reqwest::Client::new();
        let model = if self.model.contains("gpt") {
            "text-embedding-3-small"
        } else {
            &self.model
        };
        let res = client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&EmbRequest {
                input: text.to_string(),
                model: model.to_string(),
            })
            .send()
            .await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("OpenAI Embedding error: {}", err);
        }

        let data: EmbResponse = res.json().await?;
        Ok(data
            .data
            .first()
            .map(|d| d.embedding.clone())
            .unwrap_or_default())
    }
}

/// Google Gemini API provider.
pub struct GeminiProvider {
    pub api_key: String,
    pub model: String,
}
impl GeminiProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiTool>,
}

#[derive(Serialize)]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
}


#[derive(Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}


#[derive(Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_tokens: u32,
    #[serde(rename = "candidatesTokenCount")]
    completion_tokens: u32,
    #[serde(rename = "totalTokenCount")]
    total_tokens: u32,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[async_trait]
impl LlmClient for GeminiProvider {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String> {
        let client = reqwest::Client::new();
        let contents = vec![GeminiContent {
            role: "user".into(),
            parts: vec![GeminiPart {
                text: Some(prompt.to_string()),
                function_call: None,
                function_response: None,
            }],

        }];
        let req = GeminiRequest {
            contents,
            tools: vec![],
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );
        let res = client.post(&url).json(&req).send().await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("Gemini API error: {}", err);
        }

        let data: GeminiResponse = res.json().await?;
        Ok(data
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .and_then(|p| p.text.clone())
            .unwrap_or_default())
    }
}

#[async_trait]
impl ModelProvider for GeminiProvider {
    async fn complete(&self, req: Request) -> anyhow::Result<Response> {
        let client = reqwest::Client::new();
        let mut contents = Vec::new();

        for item in req.context.items.iter() {
            match item.role {
                MemoryRole::User => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: Some(item.content.clone()),
                            function_call: None,
                            function_response: None,
                        }],
                    });
                }
                MemoryRole::Assistant => {
                    let tool_calls: Option<Vec<ToolCall>> = item.metadata.get("tool_calls")
                        .and_then(|v| serde_json::from_value(v.clone()).ok());

                    let mut parts = Vec::new();
                    if !item.content.is_empty() {
                        parts.push(GeminiPart {
                            text: Some(item.content.clone()),
                            function_call: None,
                            function_response: None,
                        });
                    }

                    if let Some(tcs) = tool_calls {
                        for tc in tcs {
                            parts.push(GeminiPart {
                                text: None,
                                function_call: Some(GeminiFunctionCall {
                                    name: tc.name,
                                    args: tc.arguments,
                                }),
                                function_response: None,
                            });
                        }
                    }

                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
                MemoryRole::Tool => {
                    let tool_name = item.metadata.get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: None,
                            function_call: None,
                            function_response: Some(GeminiFunctionResponse {
                                name: tool_name,
                                response: serde_json::json!({ "result": item.content }),
                            }),
                        }],
                    });
                }
                MemoryRole::System => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: Some(format!("System Instruction: {}", item.content)),
                            function_call: None,
                            function_response: None,
                        }],
                    });
                }
            }
        }

        // --- SECONDARY SAFETY: Sanitize message sequence for Gemini ---
        let mut sanitized = Vec::new();
        let mut can_accept_tool = false;
        
        // DEBUG: Log the sequence we received
        let roles: Vec<String> = contents.iter().map(|c| c.role.clone()).collect();
        tracing::debug!("Gemini Provider: Incoming roles: {:?}", roles);

        for msg in contents {
            let has_tool_use = msg.parts.iter().any(|p| p.function_call.is_some());
            let is_tool_response = msg.parts.iter().any(|p| p.function_response.is_some());

            if has_tool_use {
                can_accept_tool = true;
            } else if is_tool_response {
                if !can_accept_tool {
                    tracing::warn!("Gemini Provider: Dropping orphaned tool response to avoid API error.");
                    continue;
                }
            } else {
                can_accept_tool = false;
            }
            sanitized.push(msg);
        }
        let mut contents = sanitized;




        contents.push(GeminiContent {
            role: "user".to_string(),
            parts: vec![GeminiPart {
                text: Some(req.prompt),
                function_call: None,
                function_response: None,
            }],

        });

        let tools = if req.tools.is_empty() {
            vec![]
        } else {
            let decls = req
                .tools
                .into_iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect();
            vec![GeminiTool {
                function_declarations: decls,
            }]
        };

        let gemini_req = GeminiRequest { contents, tools };
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let res = client.post(&url).json(&gemini_req).send().await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("Gemini API error: {}", err);
        }

        let data: GeminiResponse = res.json().await?;
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(candidate) = data.candidates.first() {
            for part in &candidate.content.parts {
                if let Some(ref t) = part.text {
                    content.push_str(t);
                }
                if let Some(ref fc) = part.function_call {
                    tool_calls.push(ToolCall {
                        name: fc.name.clone(),
                        arguments: fc.args.clone(),
                        id: uuid::Uuid::new_v4().to_string(),
                    });
                }
            }
        }

        let usage = data.usage_metadata.map(|u| ResponseUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(Response {
            content,
            tool_calls,
            usage,
        })
    }

    async fn stream_complete(
        &self,
        req: Request,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let client = reqwest::Client::new();
        let mut contents = Vec::new();

        for item in req.context.items.iter() {
            match item.role {
                MemoryRole::User => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: Some(item.content.clone()),
                            function_call: None,
                            function_response: None,
                        }],
                    });
                }
                MemoryRole::Assistant => {
                    let tool_calls: Option<Vec<ToolCall>> = item.metadata.get("tool_calls")
                        .and_then(|v| serde_json::from_value(v.clone()).ok());

                    let mut parts = Vec::new();
                    if !item.content.is_empty() {
                        parts.push(GeminiPart {
                            text: Some(item.content.clone()),
                            function_call: None,
                            function_response: None,
                        });
                    }

                    if let Some(tcs) = tool_calls {
                        for tc in tcs {
                            parts.push(GeminiPart {
                                text: None,
                                function_call: Some(GeminiFunctionCall {
                                    name: tc.name,
                                    args: tc.arguments,
                                }),
                                function_response: None,
                            });
                        }
                    }

                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
                MemoryRole::Tool => {
                    let tool_name = item.metadata.get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: None,
                            function_call: None,
                            function_response: Some(GeminiFunctionResponse {
                                name: tool_name,
                                response: serde_json::json!({ "result": item.content }),
                            }),
                        }],
                    });
                }
                MemoryRole::System => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: Some(format!("System Instruction: {}", item.content)),
                            function_call: None,
                            function_response: None,
                        }],
                    });
                }
            }
        }

        // --- SECONDARY SAFETY: Sanitize message sequence for Gemini ---
        let mut sanitized = Vec::new();
        let mut can_accept_tool = false;
        for msg in contents {
            let has_tool_use = msg.parts.iter().any(|p| p.function_call.is_some());
            let is_tool_response = msg.parts.iter().any(|p| p.function_response.is_some());

            if has_tool_use {
                can_accept_tool = true;
            } else if is_tool_response {
                if !can_accept_tool {
                    tracing::warn!("Gemini Provider: Dropping orphaned tool response to avoid API error (streaming).");
                    continue;
                }
            } else {
                can_accept_tool = false;
            }
            sanitized.push(msg);
        }
        let contents = sanitized;




        let tools = if req.tools.is_empty() {
            vec![]
        } else {
            let decls = req
                .tools
                .into_iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect();
            vec![GeminiTool {
                function_declarations: decls,
            }]
        };

        let gemini_req = GeminiRequest { contents, tools };
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}", self.model, self.api_key);

        let res = client.post(&url).json(&gemini_req).send().await?;

        let stream = async_stream::try_stream! {
            let mut byte_stream = res.bytes_stream();
            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk?;

                // Gemini stream is usually a JSON array or multiple JSON objects.
                // Basic implementation here: try to parse as Vec<GeminiResponse>
                if let Ok(event) = serde_json::from_slice::<Vec<GeminiResponse>>(&bytes) {
                    for res in event {
                        if let Some(candidate) = res.candidates.first() {
                            for part in &candidate.content.parts {
                                let mut content_delta = None;
                                let mut tool_call_delta = None;

                                if let Some(ref t) = part.text {
                                    content_delta = Some(t.clone());
                                }

                                if let Some(ref fc) = part.function_call {
                                    tool_call_delta = Some(ToolCallDelta {
                                        name: Some(fc.name.clone()),
                                        arguments_delta: Some(serde_json::to_string(&fc.args).unwrap_or_else(|_| "{}".into())),
                                        id: Some(uuid::Uuid::new_v4().to_string()),
                                    });
                                }

                                let usage = res.usage_metadata.as_ref().map(|u| ResponseUsage {
                                    prompt_tokens: u.prompt_tokens,
                                    completion_tokens: u.completion_tokens,
                                    total_tokens: u.total_tokens,
                                });

                                yield ResponseChunk { 
                                    content_delta, 
                                    tool_call_delta, 
                                    usage,
                                    is_final: false 
                                };
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
        struct GemEmbRequest {
            content: GemEmbContent,
        }
        #[derive(Serialize)]
        struct GemEmbContent {
            parts: Vec<GemEmbPart>,
        }
        #[derive(Serialize)]
        struct GemEmbPart {
            text: String,
        }
        #[derive(Deserialize)]
        struct GemEmbResponse {
            embedding: GemEmbValue,
        }
        #[derive(Deserialize)]
        struct GemEmbValue {
            values: Vec<f32>,
        }

        let client = reqwest::Client::new();
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={}", self.api_key);
        let req = GemEmbRequest {
            content: GemEmbContent {
                parts: vec![GemEmbPart {
                    text: text.to_string(),
                }],
            },
        };
        let res = client.post(&url).json(&req).send().await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            anyhow::bail!("Gemini Embedding error: {}", err);
        }

        let data: GemEmbResponse = res.json().await?;
        Ok(data.embedding.values)
    }
}

impl TokenCounter for GeminiProvider {
    fn count_tokens(&self, text: &str) -> usize {
        estimate_tokens(text)
    }
}
impl TokenCounter for OllamaProvider {
    fn count_tokens(&self, text: &str) -> usize {
        estimate_tokens(text)
    }
}
impl TokenCounter for AnthropicProvider {
    fn count_tokens(&self, text: &str) -> usize {
        estimate_tokens(text)
    }
}
impl TokenCounter for OpenAiProvider {
    fn count_tokens(&self, text: &str) -> usize {
        // For OpenAI, we fall back to a lightweight estimate for stability on Windows.
        estimate_tokens(text)
    }
}

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
#[derive(Clone)]
pub struct MemoryMetrics {
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
impl MemoryMetrics {
    pub fn new(registry: &Registry) -> anyhow::Result<Self> {
        let context_size_bytes = register_gauge_with_registry!(
            opts!("mindpalace_context_size_bytes", "desc"),
            registry
        )?;
        let item_count =
            register_gauge_with_registry!(opts!("mindpalace_item_count", "desc"), registry)?;
        let compression_ratio =
            register_gauge_with_registry!(opts!("mindpalace_compression_ratio", "desc"), registry)?;
        let layer_latency = register_histogram_with_registry!(
            "mindpalace_layer_latency_seconds",
            "desc",
            vec![0.1],
            registry
        )?;
        let fact_count =
            register_gauge_with_registry!(opts!("mindpalace_fact_count", "desc"), registry)?;
        let total_tokens_processed = register_gauge_with_registry!(
            opts!("mindpalace_total_tokens_processed", "desc"),
            registry
        )?;
        Ok(Self {
            context_size_bytes,
            item_count,
            compression_ratio,
            layer_latency,
            fact_count,
            total_tokens_processed,
        })
    }
}

pub mod config;
pub use config::MindPalaceConfig;

pub mod analysis;
pub mod metrics;

/// Utility functions for memory compression and comparison.
pub mod utils {
    /// Compresses binary data using Zstandard.
    pub fn compress(data: &[u8]) -> anyhow::Result<Vec<u8>> {
        Ok(zstd::encode_all(data, 3)?)
    }
    /// Decompresses Zstandard binary data.
    pub fn decompress(data: &[u8]) -> anyhow::Result<Vec<u8>> {
        Ok(zstd::decode_all(data)?)
    }
    /// Calculates the cosine similarity between two float vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let n_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n_a == 0.0 || n_b == 0.0 {
            0.0
        } else {
            dot / (n_a * n_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_ollama_usage_parsing() {
        let chunk = json!({
            "done": true,
            "prompt_eval_count": 10,
            "eval_count": 20
        });
        let is_done = chunk["done"].as_bool().unwrap_or(false);
        let usage = if is_done {
            let prompt_tokens = chunk["prompt_eval_count"].as_u64().unwrap_or(0) as u32;
            let completion_tokens = chunk["eval_count"].as_u64().unwrap_or(0) as u32;
            if prompt_tokens > 0 || completion_tokens > 0 {
                Some(ResponseUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                })
            } else {
                None
            }
        } else {
            None
        };
        assert!(usage.is_some());
        let u = usage.unwrap();
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 30);
    }

    #[test]
    fn test_openai_usage_parsing() {
        let data = json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40
            }
        });
        let res: OpenAiResponse = serde_json::from_value(data).unwrap();
        let usage = res.usage.map(|u| ResponseUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });
        assert!(usage.is_some());
        let u = usage.unwrap();
        assert_eq!(u.prompt_tokens, 15);
        assert_eq!(u.completion_tokens, 25);
        assert_eq!(u.total_tokens, 40);
    }

    #[test]
    fn test_gemini_usage_parsing() {
        let data = json!({
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 10,
                "totalTokenCount": 15
            }
        });
        let res: GeminiResponse = serde_json::from_value(data).unwrap();
        let usage = res.usage_metadata.map(|u| ResponseUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });
        assert!(usage.is_some());
        let u = usage.unwrap();
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 10);
        assert_eq!(u.total_tokens, 15);
    }
}
