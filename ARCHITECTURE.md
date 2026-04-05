# MindPalace Dependency DAG

MindPalace is structured using a modular layered approach. The crates compile into the `brain`, which acts as the execution engine for the `mentalist` runner. 

## Dependency Graph

```mermaid
graph TD;
    mem-core-->mem-bridge;
    mem-core-->mem-broker;
    mem-core-->mem-compactor;
    mem-core-->mem-dreamer;
    mem-core-->mem-extractor;
    mem-core-->mem-micro;
    mem-core-->mem-offloader;
    mem-core-->mem-personality;
    mem-core-->mem-resilience;
    mem-core-->mem-session;
    mem-core-->mem-retriever;

    mem-bridge-->brain;
    mem-broker-->brain;
    mem-compactor-->brain;
    mem-dreamer-->brain;
    mem-extractor-->brain;
    mem-micro-->brain;
    mem-offloader-->brain;
    mem-personality-->brain;
    mem-resilience-->brain;
    mem-session-->brain;
    mem-retriever-->brain;
    
    brain-->mentalist;
    mentalist-->gypsy;
```

## Circular Dependency Prevention
- The architecture is strictly a DAG (Directed Acyclic Graph).
- `mem-core` holds all common structs (`Context`, `MemoryItem`).
- Feature crates (`mem-*`) import `mem-core` and DO NOT import each other unless via shared traits.
- `brain` aggregates feature crates.
- We utilize Cargo's native recursive workspace checking via `cargo check --workspace` in CI which inherently fails out on circular cycles at compile time.
