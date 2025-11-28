"""
Tensor State - Enhanced with Recursive Identity Kernel

Represents Φ_k tension state with:
- Basic state management
- Gradient computation
- Recursive Identity Kernel (Δ(t) = ∇Φ - C⃗)
- Identity energy tracking
- Dimensional level detection

Mathematical Foundation:
    Δ(t) = ∇Φ(t) - C⃗(t)
    I_k(t) = eig(Δ_i(t) · Δ_j(t))
    E_id(t) = ½||Δ(t)||²
"""

import numpy as np
from typing import Optional, Tuple


class TensorState:
    """
    Enhanced Φ_k tension state with recursive identity kernel.

    This represents the core state vector that flows through collapse cycles,
    now enhanced with identity tracking and dimensional detection.
    """

    def __init__(self, values: np.ndarray, dimension: Optional[int] = None):
        """
        Initialize tensor state.

        Args:
            values: Initial Φ values (1D array)
            dimension: Optional dimension (inferred from values if not provided)
        """
        self.phi = np.array(values, dtype=np.float64)
        self.D = dimension or len(values)

        # Enhanced: Recursive Identity Kernel components
        self.curvent = np.zeros_like(self.phi)
        self.identity_kernel = np.zeros_like(self.phi)

        # History tracking for temporal operations
        self._gradient_cache = None
        self._laplacian_cache = None

    def magnitude(self) -> float:
        """
        Compute ||Φ|| (L2 norm).

        Returns:
            Tension magnitude
        """
        return np.linalg.norm(self.phi)

    def gradient(self, use_cache: bool = True) -> np.ndarray:
        """
        Compute ∇Φ (directional derivative).

        Args:
            use_cache: Whether to use cached gradient

        Returns:
            Gradient vector (same shape as phi)
        """
        if use_cache and self._gradient_cache is not None:
            return self._gradient_cache

        self._gradient_cache = np.gradient(self.phi)
        return self._gradient_cache

    def laplacian(self, use_cache: bool = True) -> np.ndarray:
        """
        Compute ∇²Φ (second derivative).

        Args:
            use_cache: Whether to use cached Laplacian

        Returns:
            Laplacian vector
        """
        if use_cache and self._laplacian_cache is not None:
            return self._laplacian_cache

        grad = self.gradient(use_cache=False)
        self._laplacian_cache = np.gradient(grad)
        return self._laplacian_cache

    def update_curvent(self, curvent: np.ndarray):
        """
        Update curvent vector C⃗(t).

        Args:
            curvent: New curvent vector
        """
        self.curvent = np.array(curvent, dtype=np.float64)

        # Invalidate caches when state changes
        self._gradient_cache = None
        self._laplacian_cache = None

    def update_identity_kernel(self) -> np.ndarray:
        """
        Compute Recursive Identity Kernel: Δ(t) = ∇Φ(t) - C⃗(t).

        This is the "delay" between where intent points and where
        the system flows, creating a unique identity signature.

        Returns:
            Identity kernel vector Δ(t)
        """
        gradient = self.gradient()
        self.identity_kernel = gradient - self.curvent
        return self.identity_kernel

    def identity_energy(self) -> float:
        """
        Compute identity energy: E_id(t) = ½||Δ(t)||².

        Returns:
            Energy stored in identity delay
        """
        return 0.5 * np.linalg.norm(self.identity_kernel) ** 2

    def identity_operator_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of identity operator: I_k(t) = eig(Δ_i · Δ_j).

        Returns:
            Eigenvalues of identity tensor (sorted descending)
        """
        # Create identity tensor as outer product
        identity_tensor = np.outer(self.identity_kernel, self.identity_kernel)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(identity_tensor)

        # Sort descending
        return np.sort(eigenvalues)[::-1]

    def detect_dimensional_level(self) -> Tuple[str, float]:
        """
        Detect which dimensional layer collapse is operating at.

        Based on the refined dimensional stack:
            0.00D: CTS permission (Φ ≈ 0)
            0.25D: Latent instability (∂Φ/∂t emerging)
            1.00D: Polarity emergence (∇Φ dominant)
            1.50D: Identity kernel (Δ(t) ≠ 0)
            2.00D: Curl recursion (requires vector field)
            3.00D: Shell stabilization (∇²Φ ≠ 0)
            3.50D: Field emission (d/dt(∇²Φ) ≠ 0)

        Returns:
            (dimension_name, confidence_score)
        """
        magnitude = self.magnitude()
        grad = self.gradient()
        grad_norm = np.linalg.norm(grad)
        lap = self.laplacian()
        lap_magnitude = np.abs(np.mean(lap))
        identity_magnitude = np.linalg.norm(self.identity_kernel)

        # Decision tree based on magnitudes
        if magnitude < 1e-6:
            return ("0.00D - CTS Permission", 1.0)

        elif identity_magnitude > 0.1 and identity_magnitude / magnitude > 0.5:
            return ("1.50D - Identity Kernel", identity_magnitude / magnitude)

        elif grad_norm > magnitude * 0.8:
            return ("1.00D - Polarity Emergence", grad_norm / magnitude)

        elif lap_magnitude > 0.1:
            return ("3.00D - Shell Stabilization", min(1.0, lap_magnitude))

        elif grad_norm > 0.01:
            return ("2.00D - Curl Recursion", 0.5)

        else:
            return ("0.25D - Latent Instability", 0.3)

    def is_stable(self, epsilon: float = 1e-6) -> bool:
        """
        Check if state has converged (collapsed).

        Args:
            epsilon: Convergence threshold

        Returns:
            True if ||Φ|| < ε
        """
        return self.magnitude() < epsilon

    def distance_to(self, other: 'TensorState') -> float:
        """
        Compute distance to another tensor state.

        Args:
            other: Another TensorState

        Returns:
            ||Φ_1 - Φ_2||
        """
        return np.linalg.norm(self.phi - other.phi)

    def copy(self) -> 'TensorState':
        """
        Create deep copy of this state.

        Returns:
            New TensorState with copied values
        """
        new_state = TensorState(self.phi.copy(), self.D)
        new_state.curvent = self.curvent.copy()
        new_state.identity_kernel = self.identity_kernel.copy()
        return new_state

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        dim_level, dim_confidence = self.detect_dimensional_level()

        return {
            'phi': self.phi.tolist(),
            'dimension': self.D,
            'magnitude': float(self.magnitude()),
            'gradient_norm': float(np.linalg.norm(self.gradient())),
            'laplacian_mean': float(np.mean(self.laplacian())),
            'curvent': self.curvent.tolist(),
            'identity_kernel': self.identity_kernel.tolist(),
            'identity_energy': float(self.identity_energy()),
            'dimensional_level': dim_level,
            'dimensional_confidence': float(dim_confidence),
            'is_stable': self.is_stable()
        }

    def __repr__(self) -> str:
        """String representation."""
        dim_level, dim_conf = self.detect_dimensional_level()
        return (f"TensorState(D={self.D}, ||Φ||={self.magnitude():.4f}, "
                f"E_id={self.identity_energy():.4f}, "
                f"level={dim_level}, conf={dim_conf:.2f})")

    def __len__(self) -> int:
        """Return dimension."""
        return self.D


# Utility functions for creating common initial states

def create_zero_state(dimension: int) -> TensorState:
    """Create zero-tension state (CTS ground state)."""
    return TensorState(np.zeros(dimension))


def create_random_state(dimension: int, scale: float = 1.0, seed: Optional[int] = None) -> TensorState:
    """
    Create random initial state.

    Args:
        dimension: State dimension
        scale: Scaling factor for random values
        seed: Random seed for reproducibility

    Returns:
        Random TensorState
    """
    if seed is not None:
        np.random.seed(seed)

    values = np.random.randn(dimension) * scale
    return TensorState(values)


def create_linear_state(dimension: int, slope: float = 1.0) -> TensorState:
    """Create linearly decreasing state."""
    values = np.linspace(slope, 0, dimension)
    return TensorState(values)


def create_oscillating_state(dimension: int, frequency: float = 1.0, amplitude: float = 1.0) -> TensorState:
    """Create sinusoidal state."""
    x = np.linspace(0, 2 * np.pi * frequency, dimension)
    values = amplitude * np.sin(x)
    return TensorState(values)


if __name__ == "__main__":
    # Demo: Create and inspect tensor states
    print("=" * 60)
    print("TENSOR STATE DEMO - Enhanced with Identity Kernel")
    print("=" * 60)

    # Create random state
    state = create_random_state(dimension=64, scale=1.0, seed=42)
    print(f"\n1. Random State:\n{state}")

    # Set some curvent
    curvent = np.random.randn(64) * 0.5
    state.update_curvent(curvent)

    # Compute identity kernel
    identity = state.update_identity_kernel()
    print(f"\n2. Identity Kernel Δ(t):")
    print(f"   ||Δ|| = {np.linalg.norm(identity):.4f}")
    print(f"   E_id = {state.identity_energy():.4f}")

    # Get eigenvalues
    eigenvalues = state.identity_operator_eigenvalues()
    print(f"\n3. Identity Operator Eigenvalues (top 5):")
    print(f"   {eigenvalues[:5]}")

    # Dimensional detection
    dim_level, confidence = state.detect_dimensional_level()
    print(f"\n4. Dimensional Level:")
    print(f"   {dim_level} (confidence: {confidence:.2f})")

    # Full state dict
    print(f"\n5. Full State Dictionary:")
    state_dict = state.to_dict()
    for key in ['magnitude', 'identity_energy', 'dimensional_level', 'is_stable']:
        print(f"   {key}: {state_dict[key]}")

    print("\n" + "=" * 60)
    print("✓ TensorState implementation complete with identity kernel")
    print("=" * 60)
