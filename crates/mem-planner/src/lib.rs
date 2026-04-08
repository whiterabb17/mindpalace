use async_trait::async_trait;
use mem_core::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub const PLAN_BOILERPLATE_TASKS: &str = "I have planned the next steps to achieve the goal.";
pub const PLAN_BOILERPLATE_NO_TASKS: &str = "I have reviewed the state and the goal is achieved. No further tasks are required.";

/// Unique identifier for a task in the execution graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TaskId(pub String);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
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
    #[serde(default)]
    pub dependencies: Vec<TaskId>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A Directed Acyclic Graph (DAG) of tasks representing the agent's plan.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionPlan {
    #[serde(deserialize_with = "deserialize_tasks")]
    pub tasks: HashMap<TaskId, TaskNode>,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub requires_approval: bool,
    #[serde(skip)]
    pub usage: Option<mem_core::ResponseUsage>,
}

fn deserialize_tasks<'de, D>(deserializer: D) -> Result<HashMap<TaskId, TaskNode>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TasksFormat {
        Map(HashMap<String, TaskNode>),
        List(Vec<TaskNode>),
        ListMap(Vec<HashMap<String, TaskNode>>),
    }

    let format = TasksFormat::deserialize(deserializer)?;
    let mut map = HashMap::new();

    match format {
        TasksFormat::Map(m) => {
            for (k, mut v) in m {
                // Ensure ID consistency
                if v.id.0.is_empty() || v.id.0 == "pending" {
                    v.id = TaskId(k.clone());
                }
                map.insert(TaskId(k), v);
            }
        }
        TasksFormat::List(l) => {
            for mut node in l {
                if node.id.0.is_empty() {
                    node.id = TaskId::new();
                }
                map.insert(node.id.clone(), node);
            }
        }
        TasksFormat::ListMap(lm) => {
            for inner_map in lm {
                for (k, mut v) in inner_map {
                    if v.id.0.is_empty() || v.id.0 == "pending" {
                        v.id = TaskId(k.clone());
                    }
                    map.insert(TaskId(k), v);
                }
            }
        }
    }

    Ok(map)
}

impl ExecutionPlan {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            content: String::new(),
            requires_approval: false,
            usage: None,
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
    /// Generates a structured execution plan based on the current context, goal, and available tools.
    async fn plan(&self, goal: &str, context: &Context, tools: Vec<mem_core::ToolDefinition>, todo_state: Option<&str>) -> anyhow::Result<ExecutionPlan>;
}

/// A production-grade LLM-driven planner.
pub struct LlmPlanner {
    client: std::sync::Arc<dyn mem_core::ModelProvider>,
}

impl LlmPlanner {
    pub fn new(client: std::sync::Arc<dyn mem_core::ModelProvider>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl PlannerEngine for LlmPlanner {
    async fn plan(&self, goal: &str, context: &Context, tools: Vec<mem_core::ToolDefinition>, todo_state: Option<&str>) -> anyhow::Result<ExecutionPlan> {
        let context_json = serde_json::to_string_pretty(context)?;
        let todo_info = todo_state.unwrap_or("No current TODO state.");
        let tools_json = serde_json::to_string_pretty(&tools)?;

        let prompt = format!(
            r#"You are the Planning Module of a Cognitive Agent. 
Your goal is to decompose a high-level objective into a Directed Acyclic Graph (DAG) of discrete tasks.

### AVAILABLE TOOLS & SKILLS ###
- **Tools**: These are executable functions (e.g. filesystem, search).
- **Skills**: These are specialized instructional tools (prefixed with 'skill:'). If a task falls into a skill's domain, you SHOULD call the skill first to retrieve its instructions.

{}

### OBJECTIVE ###
{}

### CURRENT TODO STATE ###
{}

### CONTEXT ###
{}

### INSTRUCTIONS ###
1. CHECK FOR COMPLETION: Read the CONTEXT carefully. If the user's objective is ALREADY fully satisfied by the previous information or task results, DO NOT generate any tasks.
2. If satisfied, provide a natural language summary in the "content" field and leave "tasks" empty.
3. If tasks are needed, decompose the goal into minimal necessary steps.
4. EXACT TOOL NAMES: You MUST use the EXACT tool names provided in the "AVAILABLE TOOLS & SKILLS" section. DO NOT guess or hallucinate tool names.
5. JSON FORMAT: Output the plan as a SINGLE JSON OBJECT. "tasks" should be a map where keys are task IDs and values are task nodes.
6. If a skill matches the domain (e.g. 'rust_expert' for a Rust task), prioritize calling it to get the roadmap.

Example:
{{
  "content": "Hello! How can I help you today?",
  "tasks": {{}}
}}

If the objective requires action:
{{
  "content": "I'll start by reading the main file to understand the logic.",
  "requires_approval": false,
  "tasks": {{
    "task_1": {{
      "id": "task_1",
      "name": "Read main.rs",
      "description": "Read the main.rs file",
      "tool_name": "read_file",
      "tool_args": {{ "path": "src/main.rs" }},
      "dependencies": [],
      "metadata": {{}}
    }}
  }}
}}

If the objective is complex or high-risk (e.g. data deletion, multi-step sequence):
Set "requires_approval": true.

JSON OUTPUT:
"#,
            tools_json, goal, todo_info, context_json
        );

        let req = mem_core::Request {
            prompt,
            context: std::sync::Arc::new(context.clone()),
            tools,
        };
        let response = self.client.complete(req).await?;
        let usage = response.usage.clone();
        let content = response.content;
        
        // --- PROMPT ECHO STRIPPING ---
        // Some small models repeat the prompt. We strip everything up to the first JSON brace.
        let mut clean_response = content.trim();
        if let Some(start_brace) = clean_response.find('{') {
            // Check if we have a preamble that looks like an echo
            let preamble = &clean_response[..start_brace];
            if preamble.contains("GOAL:") || preamble.contains("User:") || preamble.len() > 100 {
                tracing::info!("Detected and stripped potential prompt echo of length {}", preamble.len());
                clean_response = &clean_response[start_brace..];
            }
        }

        // Extract JSON from response (handling potential markdown fences)
        let json_str = if let Some(start) = clean_response.find('{') {
            if let Some(end) = clean_response.rfind('}') {
                &clean_response[start..=end]
            } else {
                &clean_response[start..]
            }
        } else {
            clean_response
        };

        match serde_json::from_str::<ExecutionPlan>(json_str) {
            Ok(mut plan) => {
                plan.usage = usage;
                // If content is still the full raw response, clean it up
                if plan.content.is_empty() || plan.content == content {
                    if plan.tasks.is_empty() {
                        plan.content = PLAN_BOILERPLATE_NO_TASKS.into();
                    } else {
                        plan.content = PLAN_BOILERPLATE_TASKS.into(); 
                    }
                }
                tracing::info!(tasks_count = plan.tasks.len(), "Execution plan parsed successfully");
                Ok(plan)
            }
            Err(e) => {
                tracing::warn!("Failed to parse execution plan JSON: {}. Attempting fallback recovery.", e);
                // Fallback: If it's not JSON, only use it as content if it's NOT an echo
                if content.contains(&goal[..goal.len().min(20)]) {
                     Ok(ExecutionPlan {
                        tasks: std::collections::HashMap::new(),
                        content: "I'm sorry, I encountered an error parsing the plan. Please try again.".into(),
                        requires_approval: false,
                        usage,
                    })
                } else {
                    Ok(ExecutionPlan {
                        tasks: std::collections::HashMap::new(),
                        content: content.to_string(),
                        requires_approval: false,
                        usage,
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_node_deserialization_no_metadata() {
        let json = r#"{
            "id": "task_1",
            "name": "Test Task",
            "description": "A test task",
            "dependencies": []
        }"#;
        let node: TaskNode = serde_json::from_str::<TaskNode>(json).unwrap();
        assert_eq!(node.name, "Test Task");
        assert_eq!(node.metadata, serde_json::Value::Null);
    }

    #[test]
    fn test_execution_plan_deserialization_no_content() {
        let json = r#"{
            "tasks": {
                "task_1": {
                    "id": "task_1",
                    "name": "Test Task",
                    "description": "A test task",
                    "dependencies": []
                }
            }
        }"#;
        let plan: ExecutionPlan = serde_json::from_str::<ExecutionPlan>(json).unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.content, "");
    }

    #[test]
    fn test_execution_plan_deserialization_gemini_style() {
        let json = r#"{
            "tasks": [
                {
                    "task_1": {
                        "id": "task_1",
                        "name": "Search Task",
                        "description": "Searching for bots",
                        "tool_name": "search",
                        "dependencies": []
                    }
                },
                {
                    "task_2": {
                        "id": "task_2",
                        "name": "Review Task",
                        "description": "Reviewing results",
                        "tool_name": "review",
                        "dependencies": ["task_1"]
                    }
                }
            ]
        }"#;
        let plan: ExecutionPlan = serde_json::from_str::<ExecutionPlan>(json).unwrap();
        assert_eq!(plan.tasks.len(), 2);
        assert!(plan.tasks.contains_key(&TaskId("task_1".into())));
        assert!(plan.tasks.contains_key(&TaskId("task_2".into())));
        assert_eq!(plan.tasks.get(&TaskId("task_2".into())).unwrap().dependencies[0].0, "task_1");
    }

    #[test]
    fn test_execution_plan_deserialization_list_style() {
        let json = r#"{
            "tasks": [
                {
                    "id": "task_1",
                    "name": "List Task",
                    "description": "Doing things",
                    "tool_name": "echo"
                }
            ]
        }"#;
        let plan: ExecutionPlan = serde_json::from_str::<ExecutionPlan>(json).unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert!(plan.tasks.contains_key(&TaskId("task_1".into())));
    }
}
