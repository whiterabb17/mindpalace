use mem_core::ToolDefinition;
use mem_core::db::SqliteSearchEngine;
use mem_core::FactGraph;
use serde_json::json;

/// Generates the Tool Definitions for Progressive Disclosure explicitly exposed to the AI agent.
pub fn get_progressive_disclosure_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "search_memory".to_string(),
            description: "Search local project memory with keywords to get an index of relevant observation IDs. Use this first to find context without overloading your context window.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "get_timeline".to_string(),
            description: "Get chronologically adjacent context around an observation ID.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The observation ID" }
                },
                "required": ["id"]
            }),
        },
        ToolDefinition {
            name: "get_observation".to_string(),
            description: "Fetch full detailed content for specific observation IDs.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The target observation ID" }
                },
                "required": ["id"]
            }),
        }
    ]
}

/// Dispatches standard memory tool calls.
pub fn execute_tool(
    name: &str, 
    args: &serde_json::Value,
    search_engine: &SqliteSearchEngine,
    graph: &FactGraph,
) -> anyhow::Result<String> {
    match name {
        "search_memory" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let results = search_engine.search(query)?;
            
            if results.is_empty() {
                return Ok("No relevant memory found.".to_string());
            }
            
            let mut summary = String::from("Found the following IDs:\n");
            for id in results.iter().take(5) {
                // Peek at Graph for small context snippet (simulating ~50 tokens per result)
                if let Some(fact) = graph.get_fact(id) {
                    let snippet = fact.content.chars().take(80).collect::<String>();
                    summary.push_str(&format!("- ID: {} | Snippet: {}...\n", id, snippet));
                }
            }
            summary.push_str("\n\nUse `get_observation` with the ID to fetch full details.");
            Ok(summary)
        }
        "get_observation" => {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(fact) = graph.get_fact(id) {
                let serialized = serde_json::to_string_pretty(&fact)?;
                Ok(format!("Retrieval Success:\n{}", serialized))
            } else {
                Ok("Fact not found.".to_string())
            }
        }
        "get_timeline" => {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let _results = search_engine.get_timeline(id, 5)?;
            Ok("Timeline functionality activated. (Simulation wrapper returned empty history)".to_string())
        }
        _ => Err(anyhow::anyhow!("Unknown tool name: {}", name)),
    }
}
