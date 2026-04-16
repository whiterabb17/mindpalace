use brain::Brain;
use mem_core::{Context, StorageBackend};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Represents the internal state of a circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    /// Normal operation: Requests are allowed to pass through.
    Closed,
    /// Failure threshold reached: Requests are immediately rejected.
    Open,
    /// Recovery phase: Limited requests are allowed to test for system health.
    HalfOpen,
}

/// A resilience mechanism for protecting LLM and storage backends from failure cascades.
///
/// The CircuitBreaker monitors for repeated failures, enters an `Open` state
/// to save the error budget when a threshold is tripped, and automatically
/// attempts a transition back to `Closed` after a reset timeout.
pub struct CircuitBreaker {
    /// Shared state indicator.
    pub state: Mutex<CircuitState>,
    /// Sequential failure count towards the threshold.
    pub failure_count: Mutex<usize>,
    /// Number of failures allowed before opening the circuit.
    pub failure_threshold: usize,
    /// Time to wait in the Open state before attempting a HalfOpen test.
    pub reset_timeout: Duration,
    /// Timestamp of the last recorded failure event.
    pub last_failure: Mutex<Option<Instant>>,
}

impl CircuitBreaker {
    /// Initializes a new CircuitBreaker with the specified thresholds.
    pub fn new(failure_threshold: usize, reset_timeout: Duration) -> Self {
        Self {
            state: Mutex::new(CircuitState::Closed),
            failure_count: Mutex::new(0),
            failure_threshold,
            reset_timeout,
            last_failure: Mutex::new(None),
        }
    }

    /// Determines if a request is allowed to proceed based on the circuit state.
    pub async fn can_proceed(&self) -> bool {
        let mut state = self.state.lock().await;
        if *state == CircuitState::Open {
            let last_fail = self.last_failure.lock().await;
            if let Some(instant) = *last_fail {
                // Check if the recovery timeout has elapsed.
                if instant.elapsed() > self.reset_timeout {
                    *state = CircuitState::HalfOpen;
                    return true;
                }
            }
            return false;
        }
        true
    }

    /// Signals a successful operation, potentially closing a half-open circuit.
    pub async fn report_success(&self) {
        let mut state = self.state.lock().await;
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Closed;
            let mut failures = self.failure_count.lock().await;
            *failures = 0;
        }
    }

    /// Signals an operation failure, potentially opening the circuit.
    pub async fn report_failure(&self) {
        let mut failures = self.failure_count.lock().await;
        *failures += 1;
        let mut last_fail = self.last_failure.lock().await;
        *last_fail = Some(Instant::now());

        if *failures >= self.failure_threshold {
            let mut state = self.state.lock().await;
            *state = CircuitState::Open;
        }
    }
}

/// A higher-level orchestrator adding resilience and safety logic to memory operations.
///
/// The ResilientMemoryController wraps the `Brain` and provides automatic
/// emergency snapshotting on failure, while respecting CircuitBreaker policies
/// to avoid death spirals during backend outages.
pub struct ResilientMemoryController<S: StorageBackend> {
    /// The core orchestrator instance.
    pub brain: Arc<Brain>,
    /// Backend for persisting emergency recovery state.
    pub storage: S,
    /// Shared circuit breaker for coordination across operations.
    pub circuit_breaker: Arc<CircuitBreaker>,
}

impl<S: StorageBackend> ResilientMemoryController<S> {
    /// Initializes a new ResilientMemoryController.
    pub fn new(brain: Arc<Brain>, storage: S, failure_threshold: usize) -> Self {
        Self {
            brain,
            storage,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                failure_threshold,
                Duration::from_secs(30),
            )),
        }
    }

    /// Executes memory optimization with graceful degradation and failure recovery.
    ///
    /// If the circuit is Open, optimization is skipped (saving error budget).
    /// If optimization fails, an emergency snapshot is created automatically for forensics.
    pub async fn optimize_resilient(&self, context: &mut Context) -> anyhow::Result<()> {
        if !self.circuit_breaker.can_proceed().await {
            tracing::warn!(
                "Circuit breaker OPEN. Skipping memory optimization to save error budget."
            );
            return Ok(()); // Graceful degradation.
        }

        match self.brain.optimize(context).await {
            Ok(_) => {
                self.circuit_breaker.report_success().await;
                Ok(())
            }
            Err(e) => {
                tracing::error!(
                    "Memory optimization failed: {:?}. Reporting to circuit breaker.",
                    e
                );
                self.circuit_breaker.report_failure().await;

                // Gap 9: Creating emergency snapshot on failure for recovery.
                if let Err(snapshot_err) = self.create_emergency_snapshot(context).await {
                    tracing::error!("Failed to create emergency snapshot: {:?}", snapshot_err);
                }

                Err(e)
            }
        }
    }

    /// Persists a timestamped JSON snapshot of the context into the `emergency/` directory.
    async fn create_emergency_snapshot(&self, context: &Context) -> anyhow::Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        let snapshot_id = format!("emergency/snapshot_{}.json", timestamp);
        let data = serde_json::to_vec(context)?;
        self.storage.store(&snapshot_id, &data).await?;
        tracing::info!("Emergency snapshot created: {}", snapshot_id);
        Ok(())
    }
}
