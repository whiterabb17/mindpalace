use rusqlite::{Connection, Result, params};
use std::path::PathBuf;
use crate::FactNode;
use std::sync::Mutex;

pub struct SqliteSearchEngine {
    conn: Mutex<Connection>,
}

impl SqliteSearchEngine {
    pub fn new(path: Option<PathBuf>) -> Result<Self> {
        let conn = if let Some(p) = path {
            // make sure directory exists
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            Connection::open(p)?
        } else {
            Connection::open_in_memory()?
        };
        
        // Ensure FTS5 is available (built-in with bundled)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                id UNINDEXED,
                content,
                category,
                tags
            )",
            [],
        )?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn insert_fact(&self, fact: &FactNode) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let tags_str = fact.tags.join(" ");
        conn.execute(
            "INSERT INTO memory_fts (id, content, category, tags) VALUES (?1, ?2, ?3, ?4)",
            params![fact.id, fact.content, fact.category, tags_str],
        )?;
        Ok(())
    }

    pub fn search(&self, query: &str) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id FROM memory_fts WHERE memory_fts MATCH ?1 ORDER BY rank LIMIT 20"
        )?;
        
        let mapped = stmt.query_map(params![query], |row| row.get(0))?;
        let mut results = Vec::new();
        for id in mapped {
            results.push(id?);
        }
        Ok(results)
    }
    
    pub fn get_timeline(&self, _center_id: &str, _limit: usize) -> Result<Vec<String>> {
        // Timeline feature requires querying by timestamp
        // For simplicity, we just return empty or recent items if timestamp isn't indexed in FTS.
        // In a real implementation we'd join with the main graph node table or add timestamp to FTS.
        Ok(Vec::new())
    }
}
