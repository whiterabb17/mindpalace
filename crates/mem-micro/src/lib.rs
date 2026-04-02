use mem_core::{MemoryLayer, Context};
use async_trait::async_trait;

pub struct MicroCompactor {
    pub max_tokens: usize,
}

impl Default for MicroCompactor {
    fn default() -> Self {
        Self { max_tokens: 20_000 }
    }
}

#[async_trait]
impl MemoryLayer for MicroCompactor {
    fn name(&self) -> &str {
        "MicroCompactor"
    }

    async fn process(&self, _context: &mut Context) -> anyhow::Result<()> {
        // TODO: Implement TTL-based eviction and prefix preservation
        Ok(())
    }

    fn priority(&self) -> u32 {
        2
    }
}
