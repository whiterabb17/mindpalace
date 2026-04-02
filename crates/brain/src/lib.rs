use mem_core::{MemoryLayer, Context};
use std::sync::Arc;

pub struct Brain {
    pub layers: Vec<Arc<dyn MemoryLayer>>,
}

impl Brain {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Arc<dyn MemoryLayer>) {
        self.layers.push(layer);
        self.layers.sort_by_key(|l| l.priority());
    }

    pub async fn optimize(&self, context: &mut Context) -> anyhow::Result<()> {
        for layer in &self.layers {
            layer.process(context).await?;
        }
        Ok(())
    }
}

impl Default for Brain {
    fn default() -> Self {
        Self::new()
    }
}
