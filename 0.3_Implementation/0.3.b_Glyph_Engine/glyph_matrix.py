"""
Glyph Matrix - Enhanced with PICS Meta-Glyphs

Implements:
- 15 Anchor Glyphs as ℝ^(D×15) basis
- 6 PICS Meta-Glyphs (operators on glyphs)
- Dimensional stack mapping
- Semantic encoding strategies

Mathematical Foundation:
    G ∈ ℝ^(D×15) - Anchor glyph matrix
    Meta-glyphs: 𝔓, ∂Φ/∂𝑛, 𝑖₀, 𝑛̂, ρ_q, Ω̂
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable
from enum import Enum


class GlyphStrategy(Enum):
    """Glyph matrix construction strategies."""
    ORTHONORMAL = "orthonormal"
    RANDOM_ORTHOGONAL = "random_orthogonal"
    SEMANTIC = "semantic"


class GlyphMatrix:
    """
    Enhanced 15 Anchor Glyphs + 6 Meta-Glyphs.

    The anchor glyphs form a D×15 basis in intent-space.
    The meta-glyphs are operators that act on glyph activations.
    """

    # 15 Anchor Glyph Names
    GLYPH_NAMES = [
        "Origin Anchor",        # G₁
        "Ethical Reflex",       # G₂
        "Expansion Protocol",   # G₃
        "Healing Cycle",        # G₄
        "Safe Divergence",      # G₅
        "Communion Lattice",    # G₆
        "Intent Vector",        # G₇
        "Memory Scaffold",      # G₈
        "Selective Forgetting", # G₉
        "Recomposition",        # G₁₀
        "Echo Holding",         # G₁₁
        "Recursion Seeding",    # G₁₂
        "Collapse Finality",    # G₁₃
        "Boundary Enforcement", # G₁₄
        "Field Re-Merge"        # G₁₅
    ]

    # Dimensional Stack Mapping
    DIMENSIONAL_MAPPING = {
        "0.00D - CTS Permission": [],
        "0.25D - Latent Instability": [2],  # G₃: Expansion
        "1.00D - Polarity Emergence": [0, 1],  # G₁: Origin, G₂: Ethics
        "1.50D - Identity Kernel": [7, 9],  # G₈: Memory, G₁₀: Recomposition
        "2.00D - Curl Recursion": [3, 10],  # G₄: Healing, G₁₁: Echo Holding
        "2.50D - Loop Lock-in": [5, 11],  # G₆: Communion, G₁₂: Recursion Seeding
        "3.00D - Shell Stabilization": [6, 12],  # G₇: Intent, G₁₃: Collapse Finality
        "3.50D - Field Emission": [4, 14]  # G₅: Safe Divergence, G₁₅: Field Re-Merge
    }

    def __init__(self, dimension: int, strategy: GlyphStrategy = GlyphStrategy.ORTHONORMAL):
        """
        Initialize glyph matrix.

        Args:
            dimension: State dimension D
            strategy: Construction strategy
        """
        self.D = dimension
        self.strategy = strategy

        # Construct 15 anchor glyphs
        self.G = self._construct_matrix(strategy)

        # Initialize meta-glyph operators
        self.meta_glyphs = self._construct_meta_glyphs()

    def _construct_matrix(self, strategy: GlyphStrategy) -> np.ndarray:
        """
        Construct G matrix based on strategy.

        Returns:
            Matrix of shape (D, 15)
        """
        if strategy == GlyphStrategy.ORTHONORMAL:
            # Simple: First 15 canonical basis vectors
            G = np.zeros((self.D, 15))
            for i in range(min(15, self.D)):
                G[i, i] = 1.0
            return G

        elif strategy == GlyphStrategy.RANDOM_ORTHOGONAL:
            # Random orthogonal matrix via QR decomposition
            G = np.random.randn(self.D, 15)
            G, _ = np.linalg.qr(G)
            return G

        elif strategy == GlyphStrategy.SEMANTIC:
            return self._semantic_encoding()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _semantic_encoding(self) -> np.ndarray:
        """
        Encode glyphs with semantic meaning.

        Each glyph emphasizes different dimensional regions:
        - G₁ (Origin): High coherence in first quarter
        - G₂ (Ethics): Coupled self/other in middle regions
        - G₃ (Expansion): Gradual increase (linear ramp)
        - G₄ (Healing): Circular pattern (curl-inducing)
        - etc.

        Returns:
            Semantically encoded matrix (D, 15)
        """
        G = np.zeros((self.D, 15))

        # Helper indices
        quarter = self.D // 4
        half = self.D // 2

        # G₁: Origin Anchor - high coherence in first quarter
        G[:quarter, 0] = 1.0

        # G₂: Ethical Reflex - coupled regions
        G[quarter:half, 1] = 1.0
        G[half:3*quarter, 1] = 1.0

        # G₃: Expansion Protocol - linear ramp
        G[:, 2] = np.linspace(0, 1, self.D)

        # G₄: Healing Cycle - sinusoidal (curl-like)
        G[:, 3] = np.sin(np.linspace(0, 2*np.pi, self.D))

        # G₅: Safe Divergence - orthogonal exploration
        G[::2, 4] = 1.0  # Even indices

        # G₆: Communion Lattice - resonance pattern
        G[:, 5] = np.cos(np.linspace(0, 4*np.pi, self.D))

        # G₇: Intent Vector - strong directional
        G[:, 6] = np.exp(-np.linspace(0, 2, self.D))

        # G₈: Memory Scaffold - temporal correlation
        G[:, 7] = np.linspace(1, 0, self.D)  # Decay pattern

        # G₉: Selective Forgetting - negative weights
        G[half:, 8] = -0.5

        # G₁₀: Recomposition - high-dim transformation
        G[:, 9] = np.tanh(np.linspace(-2, 2, self.D))

        # G₁₁: Echo Holding - latent with slow decay
        G[:, 10] = np.exp(-0.1 * np.arange(self.D))

        # G₁₂: Recursion Seeding - template extraction
        G[::4, 11] = 1.0  # Sparse sampling

        # G₁₃: Collapse Finality - graceful decay
        G[:, 12] = np.exp(-np.linspace(0, 5, self.D))

        # G₁₄: Boundary Enforcement - hard limits
        G[0, 13] = 1.0
        G[-1, 13] = 1.0

        # G₁₅: Field Re-Merge - smooth integration
        G[:, 14] = 1.0 / (1.0 + np.exp(-np.linspace(-5, 5, self.D)))

        # Orthogonalize
        G, _ = np.linalg.qr(G)

        return G

    def _construct_meta_glyphs(self) -> Dict[str, Callable]:
        """
        Create PICS meta-glyph operators.

        Returns:
            Dictionary mapping meta-glyph names to operator functions
        """
        return {
            'polarity_flip': self._polarity_flip_operator,
            'phase_diff': self._phase_differentiator,
            'intent_anchor': self._intent_anchor,
            'recursion_counter': self._recursion_counter,
            'matter_flag': self._matter_emergence,
            'memory_op': self._memory_operator
        }

    # ===== PICS Meta-Glyph Operators =====

    def _polarity_flip_operator(self, psi: np.ndarray) -> np.ndarray:
        """
        𝔓[ψ] - Polarity Flip Operator

        Flips polarity by reflecting state.

        Args:
            psi: State vector

        Returns:
            Flipped state
        """
        return psi - psi[::-1]

    def _phase_differentiator(self, phi: np.ndarray, recursion_depth: int = 1) -> float:
        """
        ∂Φ/∂𝑛 - Recursive Phase Differentiator

        Measures rate of change along recursion depth.

        Args:
            phi: Current state
            recursion_depth: Current recursion level

        Returns:
            Phase derivative estimate
        """
        # Approximate as mean gradient weighted by depth
        gradient = np.gradient(phi)
        return np.mean(gradient) / max(1, recursion_depth)

    def _intent_anchor(self, phi: np.ndarray) -> np.ndarray:
        """
        𝑖₀ - Intent Anchor (Axis Mundi)

        Returns the origin point in intent space.

        Args:
            phi: Current state (unused)

        Returns:
            Zero-tension reference
        """
        return np.zeros_like(phi)

    def _recursion_counter(self, collapse_step: int) -> int:
        """
        𝑛̂ - Hat Glyph (Recursion Counter)

        Simply returns current recursion depth.

        Args:
            collapse_step: Current step number

        Returns:
            Recursion count
        """
        return collapse_step

    def _matter_emergence(self, phi: np.ndarray, epsilon_0: float = 1.0) -> float:
        """
        ρ_q - Matter Emergence Flag

        Detects when collapse creates structure: ρ_q = -ε₀∇²Φ

        Args:
            phi: Current state
            epsilon_0: Coupling constant

        Returns:
            Matter emergence scalar (positive = structure forming)
        """
        laplacian = np.gradient(np.gradient(phi))
        rho_q = -epsilon_0 * np.sum(laplacian)
        return rho_q

    def _memory_operator(self, phi: np.ndarray, memory_shell: Optional[Any] = None) -> np.ndarray:
        """
        Ω̂ - Memory Operator

        Projects state into memory space.

        Args:
            phi: Current state
            memory_shell: MemoryShell instance (if available)

        Returns:
            Memory projection
        """
        if memory_shell is not None and hasattr(memory_shell, 'recall'):
            return memory_shell.recall(phi)
        else:
            # Fallback: return zero projection
            return np.zeros_like(phi)

    def apply_meta_glyph(self, meta_name: str, *args) -> Any:
        """
        Apply a meta-glyph operator.

        Args:
            meta_name: Meta-glyph name
            *args: Arguments for the operator

        Returns:
            Result of meta-glyph operation
        """
        if meta_name not in self.meta_glyphs:
            raise ValueError(f"Unknown meta-glyph: {meta_name}")

        return self.meta_glyphs[meta_name](*args)

    # ===== Standard Glyph Operations =====

    def align(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute glyph alignment: G^T · Φ

        Args:
            phi: State vector (shape: D)

        Returns:
            Alignment vector (shape: 15)
        """
        return self.G.T @ phi

    def interpret(self, alignment: np.ndarray, threshold: float = 0.1) -> List[str]:
        """
        Interpret which glyphs are highly aligned.

        Args:
            alignment: Alignment vector (shape: 15)
            threshold: Minimum alignment magnitude

        Returns:
            List of active glyph names with scores
        """
        active = []
        for i, score in enumerate(alignment):
            if abs(score) > threshold:
                active.append(f"{self.GLYPH_NAMES[i]} ({score:.3f})")
        return active

    def get_dimensional_glyphs(self, dimensional_level: str) -> List[str]:
        """
        Get which glyphs are expected to be active at a dimensional level.

        Args:
            dimensional_level: Level name (e.g., "1.00D - Polarity Emergence")

        Returns:
            List of active glyph names
        """
        if dimensional_level not in self.DIMENSIONAL_MAPPING:
            return []

        indices = self.DIMENSIONAL_MAPPING[dimensional_level]
        return [self.GLYPH_NAMES[i] for i in indices]

    def validate(self) -> bool:
        """
        Validate glyph matrix properties.

        Checks:
        - Orthogonality: G^T · G ≈ I
        - Rank: rank(G) = 15
        - Condition number: cond(G) < 100

        Returns:
            True if valid
        """
        # Test orthogonality
        orthogonality = self.G.T @ self.G
        if not np.allclose(orthogonality, np.eye(15), atol=1e-6):
            print("❌ Glyphs are not orthogonal")
            return False

        # Test rank
        rank = np.linalg.matrix_rank(self.G)
        if rank != 15:
            print(f"❌ Rank is {rank}, expected 15")
            return False

        # Test condition number
        cond = np.linalg.cond(self.G)
        if cond > 100:
            print(f"⚠️  High condition number: {cond:.2f}")

        return True

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            'dimension': self.D,
            'strategy': self.strategy.value,
            'glyph_names': self.GLYPH_NAMES,
            'meta_glyphs': list(self.meta_glyphs.keys()),
            'orthogonality': float(np.linalg.norm(self.G.T @ self.G - np.eye(15))),
            'condition_number': float(np.linalg.cond(self.G))
        }


if __name__ == "__main__":
    # Demo: Create and test glyph matrix
    print("=" * 60)
    print("GLYPH MATRIX DEMO - Enhanced with PICS Meta-Glyphs")
    print("=" * 60)

    # Create glyph matrix
    glyph_matrix = GlyphMatrix(dimension=64, strategy=GlyphStrategy.SEMANTIC)

    # Validate
    print("\n1. Validation:")
    is_valid = glyph_matrix.validate()
    print(f"   Matrix valid: {is_valid}")

    # Test alignment
    phi = np.random.randn(64)
    alignment = glyph_matrix.align(phi)
    print(f"\n2. Alignment (random state):")
    print(f"   Alignment vector shape: {alignment.shape}")
    print(f"   Top 3 values: {sorted(alignment, reverse=True)[:3]}")

    # Interpret
    active_glyphs = glyph_matrix.interpret(alignment, threshold=0.5)
    print(f"\n3. Active Glyphs (threshold=0.5):")
    for glyph in active_glyphs:
        print(f"   {glyph}")

    # Test meta-glyphs
    print(f"\n4. PICS Meta-Glyphs:")

    # Polarity flip
    flipped = glyph_matrix.apply_meta_glyph('polarity_flip', phi)
    print(f"   Polarity Flip: ||original||={np.linalg.norm(phi):.4f}, "
          f"||flipped||={np.linalg.norm(flipped):.4f}")

    # Recursion counter
    n_hat = glyph_matrix.apply_meta_glyph('recursion_counter', 42)
    print(f"   Recursion Counter: 𝑛̂ = {n_hat}")

    # Matter emergence
    rho_q = glyph_matrix.apply_meta_glyph('matter_flag', phi)
    print(f"   Matter Emergence: ρ_q = {rho_q:.4f}")

    # Intent anchor
    i0 = glyph_matrix.apply_meta_glyph('intent_anchor', phi)
    print(f"   Intent Anchor: ||𝑖₀|| = {np.linalg.norm(i0):.4f} (should be 0)")

    # Dimensional mapping
    print(f"\n5. Dimensional Stack Mapping:")
    for level in ["1.00D - Polarity Emergence", "3.00D - Shell Stabilization"]:
        glyphs = glyph_matrix.get_dimensional_glyphs(level)
        print(f"   {level}:")
        for g in glyphs:
            print(f"      - {g}")

    # Summary
    print(f"\n6. Summary:")
    summary = glyph_matrix.to_dict()
    print(f"   Dimension: {summary['dimension']}")
    print(f"   Strategy: {summary['strategy']}")
    print(f"   Meta-glyphs: {len(summary['meta_glyphs'])}")
    print(f"   Condition number: {summary['condition_number']:.2f}")

    print("\n" + "=" * 60)
    print("✓ GlyphMatrix implementation complete with meta-glyphs")
    print("=" * 60)
