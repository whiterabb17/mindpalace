use async_trait::async_trait;
use mem_core::{Context, LlmClient};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for a task in the execution graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TaskId(pub String);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

/// A single unit of work in the agent's plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: TaskId,
    pub name: String,
    pub description: String,
    pub tool_name: Option<String>,
    pub tool_args: Option<serde_json::Value>,
    pub dependencies: Vec<TaskId>,
    pub metadata: serde_json::Value,
}

/// A Directed Acyclic Graph (DAG) of tasks representing the agent's plan.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionPlan {
    pub tasks: HashMap<TaskId, TaskNode>,
}

impl ExecutionPlan {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }

    pub fn add_task(&mut self, task: TaskNode) -> TaskId {
        let id = task.id.clone();
        self.tasks.insert(id.clone(), task);
        id
    }
}

/// The core planning engine that transforms high-level goals into execution graphs.
#[async_trait]
pub trait PlannerEngine: Send + Sync {
    /// Generates a structured execution plan based on the current context and goal.
    async fn plan(&self, goal: &str, context: &Context, todo_state: Option<&str>) -> anyhow::Result<ExecutionPlan>;
}

/// A production-grade LLM-driven planner.
pub struct LlmPlanner {
    client: std::sync::Arc<dyn LlmClient>,
}

impl LlmPlanner {
    pub fn new(client: std::sync::Arc<dyn LlmClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl PlannerEngine for LlmPlanner {
    async fn plan(&self, goal: &str, context: &Context, todo_state: Option<&str>) -> anyhow::Result<ExecutionPlan> {
        let context_json = serde_json::to_string_pretty(context)?;
        let todo_info = todo_state.unwrap_or("No current TODO state.");

        let prompt = format!(
            r#"You are the Planning Module of a Cognitive Agent. 
Your goal is to decompose a high-level objective into a Directed Acyclic Graph (DAG) of discrete tasks.

### OBJECTIVE ###
{}

### CURRENT TODO STATE ###
{}

### CONTEXT ###
{}

### OUTPUT FORMAT ###
You must output a JSON object representing the ExecutionPlan.
Each task must have a unique ID, a name, a description, and a list of dependency IDs.
Crucially, if a task requires a tool, specify "tool_name" and "tool_args".

Example:
{{
  "tasks": {{
    "task_1": {{
      "id": "task_1",
      "name": "Analyze structure",
      "description": "Read the main.rs file",
      "tool_name": "read_file",
      "tool_args": {{ "path": "src/main.rs" }},
      "dependencies": []
    }},
    "task_2": {{
      "id": "task_2",
      "name": "Update Logic",
      "description": "Apply the changes",
      "tool_name": "write_file",
      "tool_args": {{ "path": "src/main.rs", "content": "..." }},
      "dependencies": ["task_1"]
    }}
  }}
}}

JSON OUTPUT:
"#,
            goal, todo_info, context_json
        );

        let response = self.client.completion(&prompt).await?;
        
        // Extract JSON from response (handling potential markdown fences)
        let json_str = if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                &response[start..=end]
            } else {
                &response[start..]
            }
        } else {
            &response
        };

        let plan: ExecutionPlan = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse execution plan: {}. Raw: {}", e, response))?;

        Ok(plan)
    }
}
