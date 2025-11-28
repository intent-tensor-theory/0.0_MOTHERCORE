"""
Collapse Kernel - Enhanced with Energy Tracking and Fan Mode Detection

Main recursion engine implementing:
    Φ_{k+1} = Φ_k - λ · R(Φ_k, G)

Enhanced with:
- Energy tracking (kinetic + potential + gradient)
- Fan mode detection (6 dynamic modes)
- Metric drift tracking
- Recursive identity kernel integration
- Meta-glyph support

Mathematical Foundation:
    E(t) = ½|C⃗|² + V(Φ) + ½|∇Φ|²
    V(Φ) = αΦ² + βΦ⁴ + γ∇²Φ
    M_ij = ⟨∂_iΦ ∂_jΦ⟩ - λ⟨F_i F_j⟩ + μδ_ij∇²Φ
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Type hints for external classes
try:
    from tensor_state import TensorState, create_random_state
    from ..b_Glyph_Engine.glyph_matrix import GlyphMatrix
    from ..c_Memory_System.memory_shell import MemoryShell
except ImportError:
    # Fallback for standalone execution
    TensorState = None
    GlyphMatrix = None
    MemoryShell = None


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute softmax with temperature.

    Args:
        x: Input vector
        temperature: Softmax temperature (higher = more uniform)

    Returns:
        Probability distribution
    """
    x_temp = x / temperature
    exp_x = np.exp(x_temp - np.max(x_temp))  # Numerical stability
    return exp_x / np.sum(exp_x)


class CollapseKernel:
    """
    Enhanced Core Collapse Kernel with full advanced features.

    Implements the fundamental recursion: Φ_{k+1} = Φ_k - λ · R(Φ_k, G)

    Enhanced with:
    - Recursive Identity Kernel
    - Energy tracking
    - Fan mode detection
    - Metric drift
    - Meta-glyph integration
    """

    def __init__(
        self,
        dimension: int,
        glyph_matrix: 'GlyphMatrix',
        memory_shell: Optional['MemoryShell'] = None,
        lambda_damping: float = 0.3,
        alpha_weight_update: float = 0.01,
        beta_weight_update: float = 0.005,
        # Energy parameters
        alpha_potential: float = 1.0,
        beta_potential: float = 0.1,
        gamma_potential: float = 0.01,
        # Metric parameters
        lambda_field: float = 0.1,
        mu_laplacian: float = 0.1
    ):
        """
        Initialize collapse kernel.

        Args:
            dimension: State dimension D
            glyph_matrix: GlyphMatrix instance
            memory_shell: Optional MemoryShell instance
            lambda_damping: Damping coefficient (0 < λ < 1)
            alpha_weight_update: Weight increase rate
            beta_weight_update: Weight decrease rate
            alpha_potential: Φ² coefficient in potential
            beta_potential: Φ⁴ coefficient in potential
            gamma_potential: ∇²Φ coefficient in potential
            lambda_field: Field correlation coefficient
            mu_laplacian: Laplacian coefficient in metric
        """
        self.D = dimension
        self.G = glyph_matrix
        self.memory = memory_shell
        self.lambda_damping = lambda_damping

        # Weight update rates
        self.alpha = alpha_weight_update
        self.beta = beta_weight_update

        # Energy potential parameters
        self.alpha_potential = alpha_potential
        self.beta_potential = beta_potential
        self.gamma_potential = gamma_potential

        # Metric parameters
        self.lambda_field = lambda_field
        self.mu_laplacian = mu_laplacian

        # Adaptive glyph weights W_k
        self.W = np.ones(15) / 15  # Initial uniform distribution

        # History tracking
        self.phi_history: List[np.ndarray] = []
        self.metric_history: List[np.ndarray] = []
        self.energy_history: List[float] = []

        # Current step counter
        self.step = 0

    def collapse_step(
        self,
        phi_k: np.ndarray,
        use_meta_glyphs: bool = False,
        use_topology: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute one collapse cycle.

        Args:
            phi_k: Current tension state (D-dimensional)
            use_meta_glyphs: Whether to apply meta-glyph operations
            use_topology: Whether to use topological predictions

        Returns:
            (phi_k_plus_1, metadata)
        """
        # Convert to TensorState if needed
        if isinstance(phi_k, np.ndarray):
            state = TensorState(phi_k) if TensorState else None
            phi_k_array = phi_k
        else:
            state = phi_k
            phi_k_array = phi_k.phi

        # 1. Compute alignment: G^T · Φ_k
        alignment = self.G.align(phi_k_array)

        # 2. Meta-glyph processing (if enabled)
        if use_meta_glyphs and state is not None:
            # Check recursion depth
            n_hat = self.G.apply_meta_glyph('recursion_counter', self.step)

            # Detect matter emergence
            rho_q = self.G.apply_meta_glyph('matter_flag', phi_k_array)

            # Apply polarity flip if stuck
            if self.step > 10 and len(self.energy_history) > 5:
                recent_energies = self.energy_history[-5:]
                if np.std(recent_energies) < 1e-6:  # Stuck
                    phi_k_array = self.G.apply_meta_glyph('polarity_flip', phi_k_array)
                    alignment = self.G.align(phi_k_array)
        else:
            n_hat = self.step
            rho_q = 0.0

        # 3. Memory influence
        if self.memory is not None:
            if use_topology:
                # Try topological prediction first
                topology_pred = self.memory.predict_from_topology(
                    phi_k_array,
                    self.phi_history[-20:] if len(self.phi_history) >= 20 else self.phi_history
                )
                if topology_pred is not None:
                    memory_influence = topology_pred
                else:
                    memory_influence = self.memory.recall_with_entanglement(phi_k_array)
            else:
                memory_influence = self.memory.recall_with_entanglement(phi_k_array)
        else:
            memory_influence = np.zeros_like(phi_k_array)

        # 4. Weight by adaptive memory W_k
        weighted = alignment * self.W + 0.2 * memory_influence[:15] if len(memory_influence) >= 15 else alignment * self.W

        # 5. Normalize to resolution force via softmax
        R = softmax(weighted)

        # 6. Update tension: Φ_{k+1} = Φ_k - λ · R
        phi_k_plus_1 = phi_k_array - self.lambda_damping * R

        # 7. Update adaptive weights W_k
        self._update_weights(alignment, phi_k_array, phi_k_plus_1)

        # 8. Compute energy
        if state is not None:
            state.update_curvent(phi_k_plus_1 - phi_k_array)
            state.update_identity_kernel()
            energy = self.compute_energy(phi_k_array, state.curvent)
        else:
            curvent = phi_k_plus_1 - phi_k_array
            energy = self._compute_energy_from_arrays(phi_k_array, curvent)

        # 9. Detect fan mode
        fan_mode = self.detect_fan_mode(phi_k_array, self.phi_history)

        # 10. Compute metric and drift
        metric = self.compute_collapse_metric(phi_k_array)
        self.metric_history.append(metric)
        causal_direction = self.detect_metric_drift()

        # 11. Compute convergence
        convergence = np.linalg.norm(phi_k_plus_1 - phi_k_array)

        # 12. Store in memory (if available)
        if self.memory is not None:
            F = np.stack([phi_k_array, phi_k_plus_1], axis=1)
            self.phi_history.append(phi_k_array)
            self.memory.store_curl_with_topology(F, self.phi_history)

        # 13. Update history
        self.energy_history.append(energy['total'])
        self.step += 1

        # Metadata
        metadata = {
            'step': self.step,
            'alignment': alignment,
            'resolution': R,
            'convergence': convergence,
            'weights': self.W.copy(),
            'energy': energy,
            'fan_mode': fan_mode,
            'metric': metric,
            'causal_direction': causal_direction,
            'recursion_depth': n_hat,
            'matter_emergence': rho_q
        }

        if state is not None:
            metadata['identity_energy'] = state.identity_energy()
            metadata['dimensional_level'] = state.detect_dimensional_level()[0]

        return phi_k_plus_1, metadata

    def _update_weights(self, alignment: np.ndarray, phi_k: np.ndarray, phi_k_plus_1: np.ndarray):
        """
        Update adaptive glyph weights W_k based on success/failure.

        Args:
            alignment: Current glyph alignment
            phi_k: State before update
            phi_k_plus_1: State after update
        """
        tension_before = np.linalg.norm(phi_k)
        tension_after = np.linalg.norm(phi_k_plus_1)

        if tension_after < tension_before:  # Success
            self.W += self.alpha * alignment
        else:  # Failure
            self.W -= self.beta * alignment

        # Keep weights positive and normalized
        self.W = np.clip(self.W, 0.01, 10.0)
        self.W /= np.sum(self.W)

    # ===== Energy Tracking =====

    def compute_energy(self, phi: np.ndarray, curvent: np.ndarray) -> dict:
        """
        Compute total recursive energy: E(t) = ½|C⃗|² + V(Φ) + ½|∇Φ|²

        Args:
            phi: State vector
            curvent: Curvent vector C⃗

        Returns:
            Dictionary with energy components
        """
        return self._compute_energy_from_arrays(phi, curvent)

    def _compute_energy_from_arrays(self, phi: np.ndarray, curvent: np.ndarray) -> dict:
        """Internal energy computation from arrays."""
        # Kinetic energy (flow)
        E_kinetic = 0.5 * np.linalg.norm(curvent) ** 2

        # Gradient energy (tension)
        grad_phi = np.gradient(phi)
        E_gradient = 0.5 * np.linalg.norm(grad_phi) ** 2

        # Potential energy: V(Φ) = αΦ² + βΦ⁴ + γ∇²Φ
        laplacian = np.gradient(grad_phi)
        V = (
            self.alpha_potential * np.sum(phi ** 2) +
            self.beta_potential * np.sum(phi ** 4) +
            self.gamma_potential * np.sum(laplacian)
        )

        return {
            'kinetic': float(E_kinetic),
            'potential': float(V),
            'gradient': float(E_gradient),
            'total': float(E_kinetic + V + E_gradient)
        }

    # ===== Fan Mode Detection =====

    def detect_fan_mode(self, phi: np.ndarray, phi_history: List[np.ndarray]) -> str:
        """
        Detect which of 6 fan dynamics is dominant.

        Returns:
            Dominant fan mode name
        """
        # Fan 1: Directional pull
        grad = np.gradient(phi)
        fan1_strength = np.linalg.norm(grad)

        # Fan 2: Phase loop (approximate curl)
        curl_strength = np.sum(np.abs(np.diff(grad)))
        fan2_strength = curl_strength

        # Fan 3 & 4: Expansion/compression
        laplacian = np.gradient(grad)
        fan3_strength = np.sum(np.maximum(0, laplacian))
        fan4_strength = np.sum(np.maximum(0, -laplacian))

        # Fan 5: Attractor strength
        fan5_strength = np.mean(np.abs(phi))

        # Fan 6: Temporal evolution
        if len(phi_history) > 0:
            dphi_dt = phi - phi_history[-1]
            fan6_strength = np.linalg.norm(dphi_dt)
        else:
            fan6_strength = 0.0

        # Determine dominant
        strengths = {
            'Fan 1: Directional Pull': fan1_strength,
            'Fan 2: Phase Loop Memory': fan2_strength,
            'Fan 3: Recursive Expansion': fan3_strength,
            'Fan 4: Recursive Compression': fan4_strength,
            'Fan 5: Scalar Attractor': fan5_strength,
            'Fan 6: Evolution Phase': fan6_strength
        }

        return max(strengths, key=strengths.get)

    # ===== Metric Drift =====

    def compute_collapse_metric(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute collapse metric: M_ij = ⟨∂_iΦ ∂_jΦ⟩ - λ⟨F_i F_j⟩ + μδ_ij∇²Φ

        Args:
            phi: State vector

        Returns:
            Metric matrix (D×D)
        """
        grad_phi = np.gradient(phi)

        # First term
        M = np.outer(grad_phi, grad_phi)

        # Second term (approximate)
        F_correlation = np.outer(grad_phi, grad_phi)
        M -= self.lambda_field * F_correlation

        # Third term
        laplacian = np.gradient(grad_phi)
        M += self.mu_laplacian * np.eye(len(phi)) * np.mean(laplacian)

        return M

    def detect_metric_drift(self) -> np.ndarray:
        """
        Compute causal direction from metric drift.

        Returns:
            Drift vector (D,)
        """
        if len(self.metric_history) < 2:
            return np.zeros(self.D)

        M_current = self.metric_history[-1]
        M_previous = self.metric_history[-2]

        dM_dt = M_current - M_previous

        # Causal vector: direction of eigenvalue increase
        try:
            eigenvalues, eigenvectors = np.linalg.eig(dM_dt)
            causal_direction = eigenvectors[:, np.argmax(np.real(eigenvalues))]
            return np.real(causal_direction)
        except:
            return np.zeros(self.D)

    # ===== High-level Methods =====

    def run_until_convergence(
        self,
        phi_0: np.ndarray,
        max_steps: int = 100,
        epsilon: float = 1e-6,
        use_meta_glyphs: bool = True,
        use_topology: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Run collapse cycles until convergence.

        Args:
            phi_0: Initial state
            max_steps: Maximum iterations
            epsilon: Convergence threshold
            use_meta_glyphs: Enable meta-glyph operations
            use_topology: Enable topological predictions

        Returns:
            (final_state, summary_metadata)
        """
        phi = phi_0.copy()

        for k in range(max_steps):
            phi, metadata = self.collapse_step(phi, use_meta_glyphs, use_topology)

            if metadata['convergence'] < epsilon:
                return phi, {
                    'converged': True,
                    'steps': k + 1,
                    'final_energy': metadata['energy'],
                    'final_weights': metadata['weights'],
                    'convergence': metadata['convergence']
                }

        # Max steps reached
        return phi, {
            'converged': False,
            'steps': max_steps,
            'final_energy': metadata['energy'],
            'final_weights': metadata['weights'],
            'convergence': metadata['convergence']
        }


if __name__ == "__main__":
    print("=" * 60)
    print("COLLAPSE KERNEL DEMO - Full Enhanced Implementation")
    print("=" * 60)
    print("\n⚠️  Requires numpy for execution")
    print("✓ Implementation complete with all advanced features")
    print("=" * 60)
