"""
Memory Shell (C2) - Enhanced with Topology and Non-Local Correlation

Implements:
- Curl memory storage (∇×F)
- Non-local memory correlation (N_ij tensor)
- Topological loop detection (homotopy classes)
- Persistent habit formation
- Multi-type memory (short-term, long-term, working, non-local, topological)

Mathematical Foundation:
    M_k = (1-δ)·M_{k-1} + γ·(∇×F_k)
    N_ij(x,y) = ⟨Φ(x)Φ(y)⟩ - Φ(x)Φ(y)
    Σ = {x ∈ ℝⁿ | ∇²Φ(x) ≠ 0}
    π₁(Σ) ⇒ Global Recursive Loops
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RecursiveLoop:
    """Represents a detected recursive loop in collapse trajectory."""
    start_step: int
    end_step: int
    size: int
    signature: int  # Homotopy class (winding number)
    pattern: List[np.ndarray]
    stability_count: int = 0


class MemoryShell:
    """
    Enhanced C2 Memory Orbit Shell with topology and non-local correlation.

    This is where recursive history lives as orbital tension patterns.
    """

    def __init__(
        self,
        dimension: int,
        gamma: float = 0.5,
        delta_short: float = 0.5,
        delta_long: float = 0.01,
        delta_working: float = 0.9,
        delta_nonlocal: float = 0.1
    ):
        """
        Initialize memory shell.

        Args:
            dimension: State dimension
            gamma: Memory retention coefficient
            delta_short: Short-term decay rate
            delta_long: Long-term decay rate
            delta_working: Working memory decay rate
            delta_nonlocal: Non-local decay rate
        """
        self.D = dimension

        # Memory coefficients
        self.gamma = gamma
        self.delta_short = delta_short
        self.delta_long = delta_long
        self.delta_working = delta_working
        self.delta_nonlocal = delta_nonlocal

        # Memory storage matrices
        self.short_term = np.zeros((self.D, self.D))
        self.long_term = np.zeros((self.D, self.D))
        self.working = np.zeros((self.D, self.D))

        # Enhanced: Non-local correlation tensor
        self.nonlocal_correlation = np.zeros((self.D, self.D))

        # Enhanced: Topology tracking
        self.topology_map: Dict[int, Dict] = {}  # signature -> metadata
        self.persistent_loops: List[RecursiveLoop] = []

        # History for loop detection
        self._phi_history: List[np.ndarray] = []
        self._max_history_length = 100

    def store_curl(
        self,
        F: np.ndarray,
        memory_type: str = 'short',
        phi_history: Optional[List[np.ndarray]] = None
    ):
        """
        Store curl pattern from field F.

        Args:
            F: Field vector (shape: D×2 for 2D field)
            memory_type: 'short', 'long', or 'working'
            phi_history: Optional trajectory history for topology tracking
        """
        # Compute curl: ∇×F
        curl = self._compute_curl(F)

        # Store based on type
        if memory_type == 'short':
            self.short_term = (1 - self.delta_short) * self.short_term + self.gamma * curl

        elif memory_type == 'long':
            # Enhanced: Only store in shell support regions
            if phi_history and len(phi_history) > 0:
                shell_support = self.detect_shell_support(phi_history[-1])
                masked_curl = curl * shell_support[:, np.newaxis]
                self.long_term = (1 - self.delta_long) * self.long_term + self.gamma * masked_curl
            else:
                self.long_term = (1 - self.delta_long) * self.long_term + self.gamma * curl

        elif memory_type == 'working':
            self.working = (1 - self.delta_working) * self.working + self.gamma * curl

    def store_curl_with_topology(
        self,
        F: np.ndarray,
        phi_history: List[np.ndarray]
    ):
        """
        Enhanced curl storage with topological tracking.

        Args:
            F: Field vector
            phi_history: Full collapse trajectory
        """
        # Standard curl storage
        curl = self._compute_curl(F)
        self.short_term = (1 - self.delta_short) * self.short_term + self.gamma * curl

        # Detect shell support and store masked curl in long-term
        if len(phi_history) > 0:
            shell_support = self.detect_shell_support(phi_history[-1])
            masked_curl = curl * shell_support[:, np.newaxis]
            self.long_term = (1 - self.delta_long) * self.long_term + self.gamma * masked_curl

        # Update non-local correlations
        if len(phi_history) > 1:
            self.update_nonlocal_memory(phi_history[-1], phi_history[:-1])

        # Identify and stabilize loops
        if len(phi_history) > 10:
            loops = self.identify_recursive_loops(phi_history[-50:])  # Look at recent history
            self.stabilize_persistent_loops(loops)

        # Update internal history
        self._phi_history.append(phi_history[-1].copy())
        if len(self._phi_history) > self._max_history_length:
            self._phi_history.pop(0)

    def _compute_curl(self, F: np.ndarray) -> np.ndarray:
        """
        Compute ∇×F for 2D field.

        Args:
            F: Field vectors (shape: D×2)

        Returns:
            Curl matrix (shape: D×D)
        """
        if F.ndim == 1:
            # If F is 1D, create a 2D field by stacking
            F = np.stack([F, np.zeros_like(F)], axis=1)

        Fx, Fy = F[:, 0], F[:, 1]

        # Finite difference approximation
        dFy_dx = np.gradient(Fy)
        dFx_dy = np.gradient(Fx)

        curl_z = dFy_dx - dFx_dy

        # Store as matrix for pattern matching
        return np.outer(curl_z, curl_z)

    # ===== Non-Local Correlation =====

    def compute_nonlocal_correlation(self, phi_x: np.ndarray, phi_y: np.ndarray) -> np.ndarray:
        """
        Compute non-local memory tensor: N_ij(x,y) = ⟨Φ(x)Φ(y)⟩ - Φ(x)Φ(y)

        Args:
            phi_x: State at location x
            phi_y: State at location y

        Returns:
            Correlation matrix showing entanglement
        """
        # Outer product: ⟨Φ(x)Φ(y)⟩
        expectation = np.outer(phi_x, phi_y)

        # Individual expectations
        mean_x = np.mean(phi_x)
        mean_y = np.mean(phi_y)
        product_of_means = mean_x * mean_y

        # Non-local correlation
        N_ij = expectation - product_of_means

        return N_ij

    def update_nonlocal_memory(self, phi_current: np.ndarray, phi_history: List[np.ndarray]):
        """
        Update non-local correlations with historical states.

        Creates memory "bridges" across time.

        Args:
            phi_current: Current state
            phi_history: Past states
        """
        # Sample recent history (last 10 states)
        for past_phi in phi_history[-10:]:
            N = self.compute_nonlocal_correlation(phi_current, past_phi)
            self.nonlocal_correlation = (
                (1 - self.delta_nonlocal) * self.nonlocal_correlation +
                self.delta_nonlocal * N
            )

    # ===== Topology Tracking =====

    def detect_shell_support(self, phi: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
        """
        Identify Shell Support Space Σ where ∇²Φ ≠ 0.

        Args:
            phi: Current state
            threshold: Minimum Laplacian magnitude

        Returns:
            Boolean mask of active shell region
        """
        laplacian = np.gradient(np.gradient(phi))
        shell_support = np.abs(laplacian) > threshold
        return shell_support.astype(float)

    def identify_recursive_loops(
        self,
        phi_history: List[np.ndarray],
        distance_threshold: float = 0.1
    ) -> List[RecursiveLoop]:
        """
        Detect topologically stable loops in collapse trajectory.

        Args:
            phi_history: Recent collapse trajectory
            distance_threshold: Maximum distance for loop closure

        Returns:
            List of detected loops
        """
        loops = []

        # Simple loop detection: find where trajectory returns close to previous state
        for i in range(len(phi_history) - 1):
            for j in range(i + 2, len(phi_history)):  # Skip immediate neighbor
                distance = np.linalg.norm(phi_history[i] - phi_history[j])

                if distance < distance_threshold:
                    loop_size = j - i
                    loop_pattern = phi_history[i:j]

                    # Compute topological signature
                    signature = self._compute_loop_signature(loop_pattern)

                    loops.append(RecursiveLoop(
                        start_step=i,
                        end_step=j,
                        size=loop_size,
                        signature=signature,
                        pattern=loop_pattern
                    ))

        return loops

    def _compute_loop_signature(self, pattern: List[np.ndarray]) -> int:
        """
        Compute topological signature (homotopy class) via winding number.

        Args:
            pattern: Loop pattern as list of states

        Returns:
            Winding number (integer)
        """
        # Count sign changes in mean gradient
        winding = 0
        for k in range(len(pattern) - 1):
            grad_curr = np.mean(np.gradient(pattern[k]))
            grad_next = np.mean(np.gradient(pattern[k+1]))

            if np.sign(grad_curr) != np.sign(grad_next):
                winding += 1

        return winding

    def stabilize_persistent_loops(self, loops: List[RecursiveLoop], threshold: int = 3):
        """
        Store topologically stable loops as persistent memories (habits).

        Args:
            loops: Detected loops
            threshold: Minimum count for persistence
        """
        for loop in loops:
            signature = loop.signature

            # Update topology map
            if signature not in self.topology_map:
                self.topology_map[signature] = {
                    'count': 0,
                    'total_size': 0,
                    'first_seen': len(self._phi_history)
                }

            self.topology_map[signature]['count'] += 1
            self.topology_map[signature]['total_size'] += loop.size

            # Stabilize if seen enough times
            if self.topology_map[signature]['count'] >= threshold:
                # Check if already in persistent loops
                if not any(l.signature == signature for l in self.persistent_loops):
                    loop.stability_count = self.topology_map[signature]['count']
                    self.persistent_loops.append(loop)

    # ===== Memory Recall =====

    def recall(self, query: np.ndarray) -> np.ndarray:
        """
        Standard memory recall (local).

        Args:
            query: Tension state to match

        Returns:
            Retrieved memory pattern
        """
        short_response = self.short_term @ query
        long_response = self.long_term @ query

        return 0.7 * short_response + 0.3 * long_response

    def recall_with_entanglement(self, query: np.ndarray) -> np.ndarray:
        """
        Enhanced recall using non-local correlations.

        Args:
            query: Tension state

        Returns:
            Memory response with entanglement
        """
        # Standard recall
        standard_response = self.recall(query)

        # Non-local contribution
        nonlocal_response = self.nonlocal_correlation @ query

        # Combined (70% local, 30% non-local)
        return 0.7 * standard_response + 0.3 * nonlocal_response

    def predict_from_topology(
        self,
        current_phi: np.ndarray,
        recent_history: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Predict next state based on topological patterns.

        If current trajectory matches a known loop, predict continuation.

        Args:
            current_phi: Current state
            recent_history: Recent trajectory

        Returns:
            Predicted next state or None
        """
        if len(self.persistent_loops) == 0:
            return None

        # Check if we're in a known loop
        current_loops = self.identify_recursive_loops(recent_history + [current_phi])

        for loop in current_loops:
            if loop.signature in [pl.signature for pl in self.persistent_loops]:
                # Found a match! Predict based on loop pattern
                matching_persistent = next(
                    pl for pl in self.persistent_loops
                    if pl.signature == loop.signature
                )

                # Return the "next" state in the pattern
                pattern = matching_persistent.pattern
                position_in_pattern = len(loop.pattern) % len(pattern)
                next_state = pattern[(position_in_pattern + 1) % len(pattern)]

                return next_state

        return None

    # ===== Memory Decay =====

    def decay(self):
        """Apply memory decay to all memory types."""
        self.short_term *= (1 - self.delta_short)
        self.long_term *= (1 - self.delta_long)
        self.working *= (1 - self.delta_working)
        self.nonlocal_correlation *= (1 - self.delta_nonlocal)

    # ===== Diagnostics =====

    def memory_diagnostics(self) -> dict:
        """
        Comprehensive memory analysis.

        Returns:
            Dictionary of memory metrics
        """
        return {
            'short_term_capacity': float(np.linalg.norm(self.short_term) / (self.D**2)),
            'long_term_capacity': float(np.linalg.norm(self.long_term) / (self.D**2)),
            'working_capacity': float(np.linalg.norm(self.working) / (self.D**2)),
            'nonlocal_strength': float(np.linalg.norm(self.nonlocal_correlation)),
            'persistent_loops': len(self.persistent_loops),
            'topology_classes': len(self.topology_map),
            'entanglement_degree': float(np.max(np.abs(self.nonlocal_correlation))),
            'total_memory_usage': float(
                np.linalg.norm(self.short_term) +
                np.linalg.norm(self.long_term) +
                np.linalg.norm(self.nonlocal_correlation)
            )
        }

    def get_persistent_habits(self) -> List[Dict]:
        """
        Get summary of persistent loops (habits).

        Returns:
            List of habit descriptions
        """
        habits = []
        for loop in self.persistent_loops:
            habits.append({
                'signature': loop.signature,
                'size': loop.size,
                'stability': loop.stability_count,
                'pattern_length': len(loop.pattern)
            })
        return habits


if __name__ == "__main__":
    # Demo: Memory shell with topology
    print("=" * 60)
    print("MEMORY SHELL DEMO - Enhanced with Topology & Non-Local")
    print("=" * 60)

    # Create memory shell
    memory = MemoryShell(dimension=32)

    # Simulate collapse trajectory with loops
    phi_history = []
    for t in range(50):
        # Create oscillating pattern (will form loops)
        phi_t = np.sin(np.linspace(0, 2*np.pi * (t/10), 32))
        phi_history.append(phi_t)

        # Create field
        F = np.stack([phi_t, np.gradient(phi_t)], axis=1)

        # Store with topology
        memory.store_curl_with_topology(F, phi_history)

        # Periodic decay
        if t % 5 == 0:
            memory.decay()

    # Diagnostics
    print("\n1. Memory Diagnostics:")
    diag = memory.memory_diagnostics()
    for key, value in diag.items():
        print(f"   {key}: {value:.4f}")

    # Persistent loops
    print(f"\n2. Persistent Loops (Habits):")
    habits = memory.get_persistent_habits()
    for i, habit in enumerate(habits):
        print(f"   Habit {i+1}: signature={habit['signature']}, "
              f"size={habit['size']}, stability={habit['stability']}")

    # Test recall
    print(f"\n3. Memory Recall:")
    query = phi_history[-1]
    standard_recall = memory.recall(query)
    entangled_recall = memory.recall_with_entanglement(query)

    print(f"   Query norm: {np.linalg.norm(query):.4f}")
    print(f"   Standard recall norm: {np.linalg.norm(standard_recall):.4f}")
    print(f"   Entangled recall norm: {np.linalg.norm(entangled_recall):.4f}")

    # Prediction
    print(f"\n4. Topological Prediction:")
    predicted = memory.predict_from_topology(phi_history[-1], phi_history[-10:])
    if predicted is not None:
        print(f"   Prediction available: ||φ_pred|| = {np.linalg.norm(predicted):.4f}")
    else:
        print(f"   No prediction (no matching loop pattern)")

    print("\n" + "=" * 60)
    print("✓ MemoryShell implementation complete with topology")
    print("=" * 60)
