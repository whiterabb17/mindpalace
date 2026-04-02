use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryRole {
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub category: String,
    pub content: String,
    pub confidence: f32,
    pub timestamp: u64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct KnowledgeBase {
    pub facts: Vec<Fact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub role: MemoryRole,
    pub content: String,
    pub timestamp: u64,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Context {
    pub items: Vec<MemoryItem>,
}

#[async_trait]
pub trait MemoryLayer: Send + Sync {
    fn name(&self) -> &str;
    async fn process(&self, context: &mut Context) -> anyhow::Result<()>;
    fn priority(&self) -> u32;
}

#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn store(&self, id: &str, data: &[u8]) -> anyhow::Result<()>;
    async fn retrieve(&self, id: &str) -> anyhow::Result<Vec<u8>>;
    async fn exists(&self, id: &str) -> bool;
    async fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>>;
}

#[derive(Clone)]
pub struct FileStorage {
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

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String>;
}

pub struct RetryLlm {
    pub inner: Arc<dyn LlmClient>,
    pub max_retries: usize,
}

impl RetryLlm {
    pub fn new(inner: Arc<dyn LlmClient>, max_retries: usize) -> Self {
        Self { inner, max_retries }
    }
}

#[async_trait]
impl LlmClient for RetryLlm {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String> {
        use tokio_retry::Retry;
        use tokio_retry::strategy::{ExponentialBackoff, jitter};

        let strategy = ExponentialBackoff::from_millis(500)
            .map(jitter)
            .take(self.max_retries);

        Retry::spawn(strategy, || self.inner.completion(prompt)).await
    }
}

pub mod utils {
    pub fn compress(data: &[u8]) -> anyhow::Result<Vec<u8>> {
        Ok(zstd::encode_all(data, 3)?)
    }

    pub fn decompress(data: &[u8]) -> anyhow::Result<Vec<u8>> {
        Ok(zstd::decode_all(data)?)
    }
}
