use async_trait::async_trait;
use serde::{Deserialize, Serialize};
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
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn completion(&self, prompt: &str) -> anyhow::Result<String>;
}

pub mod utils {
    pub fn compress(data: &[u8]) -> anyhow::Result<Vec<u8>> {
        Ok(zstd::encode_all(data, 3)?)
    }

    pub fn decompress(data: &[u8]) -> anyhow::Result<Vec<u8>> {
        Ok(zstd::decode_all(data)?)
    }
}
