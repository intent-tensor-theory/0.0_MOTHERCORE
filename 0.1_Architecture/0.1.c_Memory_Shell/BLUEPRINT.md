# 0.1.c Memory Shell (C2) — BLUEPRINT

## 🎯 Purpose

The **Memory Orbit Shell** (C2) is MOTHERCORE's 2D curl memory layer. It stores **unresolved recursion loops** (∇×F ≠ 0) until they stabilize.

This is where **recursive history** lives — not as stored data, but as **orbital tension patterns** that influence future collapse.

**C2 is the difference between a calculator and a computer.**

---

## 📐 Mathematical Foundation

### Curl Memory
```
∇×F = (∂F_y/∂x - ∂F_x/∂y)ẑ
```

When curl is non-zero, the field has **rotational memory** — it remembers its past trajectory.

### Memory Storage Equation
```
M_k = M_{k-1} + γ·(∇×F_k)
```

Where:
- **M_k** = Memory tensor at step k
- **γ** = Memory retention coefficient (0 < γ < 1)
- **∇×F_k** = Curl of current field state

Memory **decays** if not reinforced:
```
M_k = (1 - δ)·M_{k-1} + γ·(∇×F_k)
```
- **δ** = Decay rate (prevents infinite accumulation)

---

## 🏗️ Architecture

### Memory Types

| Memory Type | Storage Mechanism | Decay Rate | Purpose |
|-------------|-------------------|------------|---------|
| **Short-term** | Active curl loops | High (δ=0.5) | Current execution context |
| **Long-term** | Stabilized patterns | Low (δ=0.01) | Learned behaviors |
| **Working** | Temporary buffers | Very high (δ=0.9) | Intermediate calculations |

### Memory Operations

```python
class MemoryShell:
    """
    C2: Memory Orbit Shell

    Stores recursive history as curl patterns.
    """

    def __init__(self, dimension: int):
        self.D = dimension
        self.short_term = np.zeros((self.D, self.D))  # Curl matrix
        self.long_term = np.zeros((self.D, self.D))
        self.working = np.zeros((self.D, self.D))

    def store_curl(self, F: np.ndarray, memory_type: str = 'short'):
        """
        Store curl pattern from field F.

        Args:
            F: Field vector (shape: D×2 for 2D field)
            memory_type: 'short', 'long', or 'working'
        """
        # Compute curl: ∇×F
        curl = self._compute_curl(F)

        if memory_type == 'short':
            self.short_term = 0.5 * self.short_term + 0.5 * curl
        elif memory_type == 'long':
            self.long_term = 0.99 * self.long_term + 0.01 * curl
        elif memory_type == 'working':
            self.working = 0.1 * self.working + 0.9 * curl

    def recall(self, query: np.ndarray) -> np.ndarray:
        """
        Retrieve relevant memory based on query.

        Args:
            query: Tension state to match (shape: D)

        Returns:
            Retrieved memory pattern (shape: D)
        """
        # Weighted combination of memory types
        short_response = self.short_term @ query
        long_response = self.long_term @ query

        return 0.7 * short_response + 0.3 * long_response

    def decay(self):
        """Apply memory decay (called each step)."""
        self.short_term *= 0.5
        self.long_term *= 0.99
        self.working *= 0.1

    def _compute_curl(self, F: np.ndarray) -> np.ndarray:
        """
        Compute ∇×F for 2D field.

        Args:
            F: Field vectors (shape: D×2)

        Returns:
            Curl matrix (shape: D×D)
        """
        Fx, Fy = F[:, 0], F[:, 1]

        # Finite difference approximation
        dFy_dx = np.gradient(Fy)
        dFx_dy = np.gradient(Fx)

        curl_z = dFy_dx - dFx_dy

        # Store as matrix for pattern matching
        return np.outer(curl_z, curl_z)
```

---

## 🧪 Validation Criteria

C2 is **working** if:

1. **Storage**: Patterns are retained across multiple collapse cycles
2. **Recall**: Similar queries retrieve similar memories
3. **Decay**: Unused memories fade over time
4. **Interference**: Multiple memories can coexist without corruption
5. **Capacity**: System handles 100+ stored patterns

### Success Metrics
```python
def validate_memory_shell(memory: MemoryShell):
    # Test 1: Store and recall
    pattern = np.random.randn(memory.D, 2)
    memory.store_curl(pattern, 'short')
    query = pattern[:, 0]
    recalled = memory.recall(query)

    assert np.linalg.norm(recalled) > 0, "Memory must be retrievable"

    # Test 2: Decay
    initial_norm = np.linalg.norm(memory.short_term)
    for _ in range(10):
        memory.decay()
    final_norm = np.linalg.norm(memory.short_term)

    assert final_norm < initial_norm, "Memory must decay"

    # Test 3: Long-term stability
    memory.store_curl(pattern, 'long')
    for _ in range(100):
        memory.decay()

    assert np.linalg.norm(memory.long_term) > 0.01, "Long-term memory persists"

    return True
```

---

## 🔬 Integration with Collapse Kernel

The Memory Shell influences collapse through **memory-biased resolution**:

```python
class CollapseKernel:
    def __init__(self, ..., memory_shell: MemoryShell):
        self.memory = memory_shell

    def collapse_step(self, phi_k: np.ndarray):
        # Standard collapse
        alignment = self.G.T @ phi_k

        # Retrieve relevant memory
        memory_influence = self.memory.recall(phi_k)

        # Combine with current alignment
        weighted = alignment * self.W + 0.3 * memory_influence

        R = softmax(weighted)
        phi_k_plus_1 = phi_k - self.lambda_damping * R

        # Store current state in memory
        F = np.stack([phi_k, phi_k_plus_1], axis=1)
        self.memory.store_curl(F, 'short')

        return phi_k_plus_1, {...}
```

**This makes collapse cycles history-dependent.**

---

## 🚀 Implementation Plan

### Files to Create

```
0.1.c_Memory_Shell/
├── BLUEPRINT.md                    # This file
├── memory_shell.py                 # Main C2 class
├── curl_computer.py                # ∇×F calculation
├── pattern_matcher.py              # Memory recall logic
└── decay_manager.py                # Memory aging
```

---

## 💡 Key Insights

### Why curl for memory?
- **Rotation preserves history** — straight lines forget, circles remember
- **2D is enough** — you don't need 3D to have memory
- **Natural decay** — curl dissipates without reinforcement

### Why three memory types?
- **Short-term** — current execution (like CPU registers)
- **Long-term** — learned patterns (like hard disk)
- **Working** — scratch space (like cache)

### How does this relate to human memory?
- **Exact match:** Short-term memory (what was I just doing?)
- **Similar match:** Long-term memory (this reminds me of...)
- **Rapid forgetting:** Working memory (temporary calculations)

---

## 📊 Expected Behavior

After 100 collapse cycles:
- Common patterns strengthen in long-term memory
- Recent patterns dominate short-term memory
- Rarely-used patterns fade completely
- **System develops "preferences" based on history**

**This is the beginning of learned behavior.**

---

**Status:** 🔵 BLUEPRINT — Architecture defined, implementation pending.

**Next:** Create `0.3.c_Memory_System/` implementation files.
