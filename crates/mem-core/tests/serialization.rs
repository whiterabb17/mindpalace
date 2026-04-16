use mem_core::{FactGraph, FactNode};

#[test]
fn test_fact_graph_serialization_roundtrip() {
    let graph = FactGraph::new(None).unwrap();
    let fact = FactNode::new(
        "The capital of France is Paris".to_string(),
        "Geography".to_string(),
        1.0,
        "session-1".to_string(),
    );
    let fact_id = fact.id.clone();
    graph.add_fact(fact).unwrap();

    let serialized = serde_json::to_string(&graph).unwrap();
    println!("Serialized: {}", serialized);

    let deserialized: FactGraph = serde_json::from_str(&serialized).unwrap();
    let recovered_fact = deserialized
        .get_fact(&fact_id)
        .expect("Fact not found after deserialization");

    assert_eq!(recovered_fact.content, "The capital of France is Paris");
    assert_eq!(recovered_fact.category, "Geography");
}

#[test]
fn test_fact_graph_deserialization_empty() {
    let empty_json = "[]";
    let deserialized: FactGraph = serde_json::from_str(empty_json).unwrap();
    assert!(deserialized.all_active_facts().is_empty());
}
