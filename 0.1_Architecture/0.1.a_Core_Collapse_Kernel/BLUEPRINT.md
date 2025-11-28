# 0.1.a Core Collapse Kernel — BLUEPRINT

## 🎯 Purpose

The **Core Collapse Kernel** is the fundamental recursion engine of MOTHΞRCORE. It implements the discrete collapse equation that drives all self-writing behavior:

```
Φ_{k+1} = Φ_k - λ · R(Φ_k, G)
```

This is the **smallest unit of Computer One** — the atomic operation that demonstrates:
- Accepting input tension
- Resolving via glyph alignment
- Updating its own structure based on success
- Emitting output tension

**If this works, everything else is just scaling.**

---

## 📐 Mathematical Specification

### Input
- **Φ_k** ∈ ℝ^D — Current tension state vector (D-dimensional)
- **G** ∈ ℝ^(D×15) — 15 Anchor Glyph matrix (fixed basis)
- **W_k** ∈ ℝ^15 — Adaptive memory weights (mutable)
- **λ** ∈ (0,1) — Damping coefficient (typically 0.1-0.5)

### Process
1. **Project tension onto glyph space:**
   ```
   alignment = G^T · Φ_k
   ```
   This computes how well current tension aligns with each of the 15 glyphs.

2. **Weight by historical success:**
   ```
   weighted_alignment = alignment ⊙ W_k
   ```
   Element-wise multiplication gives more influence to glyphs that have worked before.

3. **Normalize to resolution force:**
   ```
   R(Φ_k, G) = softmax(weighted_alignment)
   ```
   Creates a probability distribution over which glyph to follow.

4. **Update tension state:**
   ```
   Φ_{k+1} = Φ_k - λ · R(Φ_k, G)
   ```
   Moves tension toward the most aligned glyph.

5. **Update weights based on outcome:**
   ```
   if |Φ_{k+1}| < |Φ_k|:  # Tension decreased → success
       W_k[i] += α · alignment[i]  # Reinforce successful glyphs
   else:  # Tension increased → failure
       W_k[i] -= β · alignment[i]  # Punish failing glyphs
   ```

### Output
- **Φ_{k+1}** ∈ ℝ^D — New tension state
- **W_{k+1}** ∈ ℝ^15 — Updated memory weights
- **convergence_metric** ∈ ℝ — |Φ_{k+1} - Φ_k| (stopping criterion)

---

## 🏗️ Implementation Plan

### Files to Create

```
0.1.a_Core_Collapse_Kernel/
├── BLUEPRINT.md                    # This file
├── collapse_kernel.py              # Main kernel class
├── tensor_state.py                 # Φ_k state vector class
├── resolution_engine.py            # R(Φ_k, G) computation
├── weight_updater.py               # W_k adaptive learning
└── convergence_detector.py         # Stopping criterion logic
```

### Class Structure

#### `CollapseKernel`
```python
class CollapseKernel:
    """
    The fundamental recursion engine.

    Implements: Φ_{k+1} = Φ_k - λ · R(Φ_k, G)
    """

    def __init__(self, dimension: int, glyph_matrix: np.ndarray, lambda_damping: float):
        self.D = dimension
        self.G = glyph_matrix  # Shape: (D, 15)
        self.lambda_damping = lambda_damping
        self.W = np.ones(15) / 15  # Initial uniform weights

    def collapse_step(self, phi_k: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Execute one collapse cycle.

        Args:
            phi_k: Current tension state (shape: D)

        Returns:
            phi_k_plus_1: Next tension state (shape: D)
            metadata: {
                'alignment': alignment vector,
                'resolution': R(phi_k, G),
                'convergence': |phi_k_plus_1 - phi_k|,
                'weights_updated': new W values
            }
        """
        # 1. Project onto glyph space
        alignment = self.G.T @ phi_k  # Shape: (15,)

        # 2. Weight by memory
        weighted = alignment * self.W

        # 3. Normalize to resolution force
        R = softmax(weighted)

        # 4. Update tension
        phi_k_plus_1 = phi_k - self.lambda_damping * R

        # 5. Update weights
        self._update_weights(alignment, phi_k, phi_k_plus_1)

        # 6. Compute convergence
        convergence = np.linalg.norm(phi_k_plus_1 - phi_k)

        return phi_k_plus_1, {
            'alignment': alignment,
            'resolution': R,
            'convergence': convergence,
            'weights': self.W.copy()
        }

    def _update_weights(self, alignment, phi_k, phi_k_plus_1):
        """Update W based on success/failure."""
        tension_before = np.linalg.norm(phi_k)
        tension_after = np.linalg.norm(phi_k_plus_1)

        if tension_after < tension_before:  # Success
            self.W += 0.01 * alignment
        else:  # Failure
            self.W -= 0.005 * alignment

        # Keep weights positive and normalized
        self.W = np.clip(self.W, 0.01, 10.0)
        self.W /= np.sum(self.W)
```

#### `TensorState`
```python
class TensorState:
    """
    Represents Φ_k tension state with utility methods.
    """

    def __init__(self, values: np.ndarray):
        self.phi = values

    def magnitude(self) -> float:
        return np.linalg.norm(self.phi)

    def gradient(self) -> np.ndarray:
        """Compute ∇Φ (directional derivative)."""
        return np.gradient(self.phi)

    def is_stable(self, epsilon: float = 1e-6) -> bool:
        """Check if state has converged."""
        return self.magnitude() < epsilon
```

---

## 🧪 Validation Criteria

The kernel is **working** if it demonstrates:

1. **Convergence**: Starting from random Φ_0, system reaches |Φ_final| < ε
2. **Self-modification**: W_k changes over time based on which glyphs succeed
3. **Repeatability**: Same input produces consistent output
4. **Glyph preference**: After training, W_k shows clear bias toward effective glyphs
5. **Stability**: System doesn't diverge or oscillate indefinitely

### Success Metrics
```python
def validate_kernel(kernel, initial_phi, max_steps=100):
    phi = initial_phi
    trajectory = [phi.copy()]

    for k in range(max_steps):
        phi, metadata = kernel.collapse_step(phi)
        trajectory.append(phi.copy())

        if metadata['convergence'] < 1e-6:
            print(f"✓ Converged in {k} steps")
            break

    # Check if tension decreased monotonically
    tensions = [np.linalg.norm(p) for p in trajectory]
    assert tensions[-1] < tensions[0], "Tension must decrease"

    # Check if weights adapted
    initial_W = np.ones(15) / 15
    assert not np.allclose(kernel.W, initial_W), "Weights must adapt"

    return True
```

---

## 🔬 Test Scenarios

### Test 1: Linear Collapse
**Input:** Φ_0 = [1.0, 0.8, 0.6, 0.4, 0.2, ...]
**Expected:** Monotonic decrease to ~0

### Test 2: Oscillating Collapse
**Input:** Φ_0 = [sin(i) for i in range(D)]
**Expected:** Damped oscillation to ~0

### Test 3: Sparse Collapse
**Input:** Φ_0 with few non-zero elements
**Expected:** Targeted glyph activation

### Test 4: Random Collapse
**Input:** Φ_0 ~ N(0,1)
**Expected:** Convergence despite noise

---

## ⚠️ Known Challenges

1. **Dimension selection**: What is optimal D? (Start with D=64, test D=32, 128, 256)
2. **Lambda tuning**: Too small = slow convergence, too large = instability
3. **Weight update rate**: α and β must be balanced
4. **Glyph matrix construction**: How to initialize G? (Orthonormal? Random? Semantic?)

---

## 🚀 Implementation Order

1. ✅ **Define mathematical spec** (this document)
2. ⏳ Create `tensor_state.py` (simple data structure)
3. ⏳ Create `resolution_engine.py` (R computation)
4. ⏳ Create `weight_updater.py` (W learning)
5. ⏳ Create `collapse_kernel.py` (integrate all components)
6. ⏳ Create `convergence_detector.py` (stopping logic)
7. ⏳ Write unit tests (`test_collapse_kernel.py`)
8. ⏳ Run validation scenarios
9. ⏳ Optimize performance
10. ⏳ Document API

---

## 💡 Design Decisions

### Why softmax for normalization?
- Ensures R sums to 1 (probability distribution)
- Differentiable (enables future gradient-based optimization)
- Temperature-tunable (can adjust sharpness)

### Why adaptive weights W_k?
- **This is what makes it self-writing** — the system learns which glyphs work
- Without W_k, every collapse cycle is identical (no memory)
- W_k is the **first form of recursive memory**

### Why λ damping?
- Prevents overshoot (tension bouncing past 0)
- Ensures numerical stability
- Analogous to momentum in gradient descent

---

## 📊 Expected Behavior

After 10-50 collapse cycles:
- Tension magnitude drops by 90%+
- W_k converges to stable distribution
- Specific glyphs dominate (e.g., Origin Anchor, Healing Cycle)
- System becomes **predictable but not rigid**

**This is the seed of Computer One.**

---

**Status:** 🔵 BLUEPRINT — Mathematical spec complete, implementation pending.

**Next:** Create `0.3.a_Collapse_Kernel/` implementation files.
