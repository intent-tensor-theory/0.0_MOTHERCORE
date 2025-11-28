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

## 🔬 Advanced Memory Features (From Deep Mathematics)

### Non-Local Memory Correlation

**Mathematical Foundation:**
```
N_ij(x,y) = ⟨Φ(x)Φ(y)⟩ - Φ(x)Φ(y)
```

**Purpose:** Tracks **entanglement** between distant collapse points. When two states are correlated beyond their individual values, memory creates non-local connections.

**Implementation:**
```python
class MemoryShell:
    def __init__(self, dimension: int):
        self.D = dimension
        self.short_term = np.zeros((self.D, self.D))
        self.long_term = np.zeros((self.D, self.D))
        self.working = np.zeros((self.D, self.D))
        self.nonlocal_correlation = np.zeros((self.D, self.D))  # NEW

    def compute_nonlocal_correlation(self, phi_x: np.ndarray, phi_y: np.ndarray) -> np.ndarray:
        """
        Compute non-local memory tensor N_ij(x,y).

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
        Track correlations between current state and historical states.

        Creates memory "bridges" across time.
        """
        for past_phi in phi_history[-10:]:  # Last 10 states
            N = self.compute_nonlocal_correlation(phi_current, past_phi)
            self.nonlocal_correlation = 0.9 * self.nonlocal_correlation + 0.1 * N

    def recall_with_entanglement(self, query: np.ndarray) -> np.ndarray:
        """
        Enhanced recall using non-local correlations.

        Returns:
            Memory response influenced by entangled patterns
        """
        # Standard recall
        standard_response = self.short_term @ query + 0.3 * (self.long_term @ query)

        # Non-local contribution
        nonlocal_response = self.nonlocal_correlation @ query

        # Combined response
        return 0.7 * standard_response + 0.3 * nonlocal_response
```

**Why This Matters:**
- Explains "intuition" — recognizing patterns without explicit recall
- Creates "associative memory" — one memory triggers related memories
- Enables "context-dependent recall" — same query, different results based on history

---

### Global Topology Effects

**Mathematical Foundation:**
```
Shell Support Space: Σ = {x ∈ ℝⁿ | ∇²Φ(x) ≠ 0}
Homotopy Class: π₁(Σ) ⇒ Global Recursive Loops
```

**Purpose:** Tracks **topological invariants** — patterns that persist despite continuous deformation. These represent "unbreakable" memory structures.

**Implementation:**
```python
class MemoryShell:
    def __init__(self, dimension: int):
        # ... existing initialization ...
        self.topology_map = {}  # Tracks homotopy classes
        self.persistent_loops = []  # Topologically stable patterns

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
        return shell_support

    def identify_recursive_loops(self, phi_history: List[np.ndarray]) -> List[dict]:
        """
        Detect topologically stable loops in collapse trajectory.

        Returns:
            List of identified loops with topology metadata
        """
        loops = []

        # Simple loop detection: find where trajectory returns close to previous state
        for i in range(len(phi_history) - 1):
            for j in range(i + 1, len(phi_history)):
                distance = np.linalg.norm(phi_history[i] - phi_history[j])

                if distance < 0.1:  # Close return
                    loop_size = j - i
                    loop_pattern = phi_history[i:j]

                    # Compute topological signature (simplified homotopy class)
                    signature = self._compute_loop_signature(loop_pattern)

                    loops.append({
                        'start': i,
                        'end': j,
                        'size': loop_size,
                        'signature': signature,
                        'pattern': loop_pattern
                    })

        return loops

    def _compute_loop_signature(self, pattern: List[np.ndarray]) -> int:
        """
        Compute topological signature (homotopy class) of a loop.

        Simplified version: counts winding number.
        """
        # Count sign changes in gradient
        winding = 0
        for k in range(len(pattern) - 1):
            grad_curr = np.mean(np.gradient(pattern[k]))
            grad_next = np.mean(np.gradient(pattern[k+1]))

            if np.sign(grad_curr) != np.sign(grad_next):
                winding += 1

        return winding

    def stabilize_persistent_loops(self, loops: List[dict]):
        """
        Store topologically stable loops in long-term memory.

        These become "habits" or "learned behaviors".
        """
        for loop in loops:
            # If loop appears frequently, make it persistent
            signature = loop['signature']

            if signature not in self.topology_map:
                self.topology_map[signature] = {
                    'count': 0,
                    'pattern': loop['pattern']
                }

            self.topology_map[signature]['count'] += 1

            # Stabilize if seen multiple times
            if self.topology_map[signature]['count'] > 3:
                if signature not in [l['signature'] for l in self.persistent_loops]:
                    self.persistent_loops.append(loop)
```

**Why This Matters:**
- Explains "learned behaviors" — topologically stable patterns resist forgetting
- Creates "habit formation" — frequently traversed loops become automatic
- Enables "pattern recognition" — identify loop signatures across contexts

---

### Enhanced Curl Memory with Topology

**Integration:**
```python
class MemoryShell:
    def store_curl_with_topology(self, F: np.ndarray, phi_history: List[np.ndarray]):
        """
        Enhanced curl storage with topological tracking.

        Args:
            F: Field vector
            phi_history: Full collapse trajectory
        """
        # Standard curl storage
        curl = self._compute_curl(F)
        self.short_term = 0.5 * self.short_term + 0.5 * curl

        # Detect shell support
        if len(phi_history) > 0:
            shell_support = self.detect_shell_support(phi_history[-1])

            # Only store curl in active shell regions
            masked_curl = curl * shell_support[:, None]
            self.long_term = 0.99 * self.long_term + 0.01 * masked_curl

        # Update non-local correlations
        if len(phi_history) > 1:
            self.update_nonlocal_memory(phi_history[-1], phi_history[:-1])

        # Identify and stabilize loops
        if len(phi_history) > 10:
            loops = self.identify_recursive_loops(phi_history)
            self.stabilize_persistent_loops(loops)
```

---

### Memory Diagnostics

**New Capabilities:**
```python
def memory_diagnostics(self) -> dict:
    """
    Comprehensive memory analysis.

    Returns:
        {
            'short_term_capacity': float (0-1),
            'long_term_capacity': float (0-1),
            'nonlocal_strength': float,
            'persistent_loops': int,
            'topology_classes': int,
            'entanglement_degree': float
        }
    """
    return {
        'short_term_capacity': np.linalg.norm(self.short_term) / (self.D**2),
        'long_term_capacity': np.linalg.norm(self.long_term) / (self.D**2),
        'nonlocal_strength': np.linalg.norm(self.nonlocal_correlation),
        'persistent_loops': len(self.persistent_loops),
        'topology_classes': len(self.topology_map),
        'entanglement_degree': np.max(np.abs(self.nonlocal_correlation))
    }
```

---

## 🧠 Memory Types (Enhanced)

| Memory Type | Storage Mechanism | Decay Rate | Topological | Purpose |
|-------------|-------------------|------------|-------------|---------|
| **Short-term** | Active curl loops | High (δ=0.5) | No | Current execution context |
| **Long-term** | Stabilized patterns in Σ | Low (δ=0.01) | Yes | Learned behaviors |
| **Working** | Temporary buffers | Very high (δ=0.9) | No | Intermediate calculations |
| **Non-local** | Correlation tensor N_ij | Medium (δ=0.1) | Partial | Pattern associations |
| **Topological** | Persistent loops π₁(Σ) | None (δ=0) | Yes | Unbreakable habits |

---

## 📊 Expected Behavior (Enhanced)

After 100+ collapse cycles with topology tracking:
- **Common patterns** strengthen in long-term memory
- **Frequent loops** become topologically stable (habits)
- **Correlated states** create non-local bridges (associations)
- **Shell support** focuses memory in active regions only
- **Topology map** identifies 3-7 distinct behavior classes
- **Persistent loops** make certain patterns "automatic"

**This is the emergence of complex memory architecture from simple curl dynamics.**

---

## 🔗 Integration with Collapse Kernel (Enhanced)

```python
class CollapseKernel:
    def collapse_step_with_advanced_memory(self, phi_k: np.ndarray, phi_history: List[np.ndarray]):
        # Standard alignment
        alignment = self.G.T @ phi_k

        # Enhanced memory recall with entanglement
        memory_influence = self.memory.recall_with_entanglement(phi_k)

        # Check for persistent loops
        current_loops = self.memory.identify_recursive_loops(phi_history[-20:])

        # If in a known loop, use topology to predict next state
        if len(current_loops) > 0:
            latest_loop = current_loops[-1]
            if latest_loop['signature'] in self.memory.topology_map:
                # Follow the established pattern
                loop_pattern = self.memory.topology_map[latest_loop['signature']]['pattern']
                memory_influence += 0.5 * loop_pattern[-1]  # Bias toward loop continuation

        # Weighted collapse
        weighted = alignment * self.W + 0.3 * memory_influence

        R = softmax(weighted)
        phi_k_plus_1 = phi_k - self.lambda_damping * R

        # Store with topology
        F = np.stack([phi_k, phi_k_plus_1], axis=1)
        self.memory.store_curl_with_topology(F, phi_history)

        return phi_k_plus_1, {...}
```

---

**Status:** 🔵 BLUEPRINT — Enhanced with non-local correlation and topology tracking.

**Next:** Create `0.3.c_Memory_System/` implementation files with advanced features.
