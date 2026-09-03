# Deterministic SLM Inference on CPU

An empirical verification harness demonstrating that autoregressive token decoding in Small Language Models (SLMs) is a closed, bit-level deterministic algebraic computation when isolated under strict single-process CPU constraints.

---

## Executive Summary

When stochastic sampling is disabled (`do_sample=False`, `num_beams=1`), internal thread concurrency is clamped, and batch size is fixed to an isolated single request ($B = 1$), an autoregressive language model behaves as a pure mathematical function:

$$f(X; \theta) = \mathbf{z} \implies x_t = \arg\max_{i \in \mathcal{V}} (z_i)$$

Under these boundary conditions, generating tokens across 100 sequential passes on the same physical CPU produces an identical SHA-256 byte-level cryptographic digest with **zero bit-flips**.

---

## Theoretical Framework: Three Orthogonal Layers

To prevent misleading claims regarding model reliability or enterprise readiness, this repository strictly decouples three layers that are frequently conflated:

```
+-------------------------------------------------------------------------+
| Layer 1: Hardware & Algebraic Execution (Computational Determinism)    |
| -> Invariant linear algebra: Y = WX + b under fixed reduction trees.    |
| -> VERIFIED: Bit-for-bit identical across 100 iterations.              |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
| Layer 2: Knowledge & Semantic Representation (Statistical Epistemology) |
| -> Model weights (θ) are lossy statistical compressions of web corpora. |
| -> UNCHANGED: Prone to hallucination, sycophancy, and logical failure.  |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
| Layer 3: Enterprise Serving Architecture (Production Feasibility)       |
| -> B=1 greedy decoding on CPU is commercially and operationally unviable|
| -> REQUIRES: External guards (AST parsers, schemas, sandboxing, evals). |
+-------------------------------------------------------------------------+
```

---

## What This Proves vs. What It Explicitly Does NOT Prove

### What This Proves

* **Local Algorithmic Invariance:** Transformers contain no intrinsic randomness during inference. The forward pass is entirely deterministic given fixed inputs, frozen parameters ($\theta$), and an invariant compute graph.
* **Elimination of Hardware Race Conditions:** By enforcing strict OpenMP/BLAS thread clamping (`torch.set_num_threads(2)`) on CPU architectures, non-associative floating-point summation trees produce identical numerical outputs across independent runs.
* **Cryptographic Token Reproducibility:** Every generated token ID in the sequence matches across $N = 100$ iterations, yielding zero byte drift in the output tensor buffer.

### What This Does NOT Prove (Rigid Boundaries & Non-Goals)

* **Does NOT Prove Model Truthfulness or Reliability (Deterministic Hallucination):**  
  A deterministic system guarantees only **reproducibility of output**, not **validity of content**. If the greedy decoding path ($\arg\max$) selects an incorrect factual assertion, the model will reproduce that identical falsehood with 100% certainty across every single pass.
* **Does NOT Represent a Production Serving Architecture (Zero Generalization Claim):**  
  * **Throughput Infeasibility ($B = 1$):** Real-world serving systems rely on dynamic continuous batching ($B > 1$), speculative decoding, and asynchronous scheduling (e.g., vLLM, TGI). These enterprise techniques introduce tensor memory interleaving and non-deterministic operation scheduling to maximize GPU utilization.
  * **Creativity Mode Collapse:** Clamping $T = 0$ (`do_sample=False`) is useful for code and structured JSON extraction, but causes mode collapse, catastrophic repetition, and degraded linguistic diversity in open-ended conversational tasks.
* **Does NOT Guarantee Cross-Hardware Hash Equivalence:**  
  The cryptographic digest is invariant **strictly within the same CPU microarchitecture and compiler toolchain**. Executing this same repository on differing instruction sets (e.g., Intel AVX2 vs. AMD AVX-512 vs. Apple Silicon ARM NEON) can lead to minor rounding variances in floating-point operations ($10^{-7}$). If two tokens are separated by an infinitesimal logit margin, this variation can trigger a logit flip without invalidating the intra-hardware determinism of either machine.

---

## Empirical Reproduction Proof

The following empirical ground-truth was generated on an isolated CPU test environment:

* **Hardware Architecture:** Intel Core Processor (x86_64, Dual-Thread Clamped)
* **Operating System:** Ubuntu 24.04 LTS (Linux 6.8 Kernel)
* **Model Checkpoint:** `SmolLM2-135M-Instruct` (Pre-Quantized Q4_K_M GGUF, ~105 MB)
* **Inference Harness:** Single Process, Batch Size $B = 1$, Greedy ArgMax Decoding

```text
=================================================================
VERIFICATION STATUS: PASSED (100% Deterministic)
Iterations Checked : 100 / 100 identical passes
Model Target       : SmolLM2-135M-Instruct-Q4_K_M.gguf (~105 MB)
SHA-256 Digest     : eefda54ad8c341911b77331baefcf0e2388167c905e6d7070d4094c1bc04eb20
Prompt Input       : "Explain in exactly one sentence what causes a race condition in concurrent computing."
Canonical Output   : "In concurrent computing, a race condition occurs when two or more threads or processes attempt to access the same data simultaneously, leading to"
=================================================================
```

---

## System Constraints & Invalidation Matrix

This experiment remains valid **if and only if** every constraint in the matrix below is rigorously enforced:

| Invalidation Boundary | Operational Setting | Root Cause of Failure if Violated |
| :--- | :--- | :--- |
| **Sampling Mechanism** | `do_sample=False`, `num_beams=1` | Enables pseudo-random categorical drawing (Multinomial Softmax) across the token logit distribution. |
| **Batch Concurrency** | Batch Size $B = 1$, Isolated Process | Dynamic batching interleaves attention key/value caches, modifying memory strides and intermediate tensor alignments. |
| **Thread Scheduling** | `torch.set_num_threads(2)` | Unbounded multi-core thread scheduling alters the order of associative floating-point additions ($a + (b + c) \neq (a + b) + c$ in IEEE 754 float precision). |
| **Execution State** | `model.eval()`, `torch.inference_mode()` | Unfrozen evaluation states leave stochastic mechanisms (e.g., Dropout masks) active in the forward pass. |
| **Platform Target** | CPU-Only PyTorch Wheel (`whl/cpu`) | Standard GPU execution pipelines dispatch asynchronous CUDA kernels where warp schedulers do not guarantee deterministic order of reduction. |
| **Memory Allocation** | Static Pre-Quantized GGUF Loading | Dynamic in-memory quantization generates on-the-fly rounding artifacts; using pre-quantized static weights guarantees invariant weight offsets. |


## Quickstart & Local Reproduction

Verify bit-level computational determinism locally in under 60 seconds on any consumer laptop (no GPU or CUDA configuration required).

### Prerequisites

* Linux / macOS / Windows WSL2
* Python >= 3.12
* [`uv`](https://docs.astral.sh/uv/) (Extremely fast Python package and project manager)
* `curl` (for binary checkpoint retrieval)

### Reproduction Steps

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/deterministic-llm-cpu.git
cd deterministic-llm-cpu

# 2. Synchronize isolated CPU-only virtual environment
make install

# 3. Verify absolute absence of CUDA / GPU runtime dependencies
make test

# 4. Execute the automated 100-iteration evaluation harness
make run
```

---

## Dual Execution Modalities

This repository uses a dual-surface architecture via `marimo` and `app.py`, allowing the exact same codebase to serve both automated pipelines and interactive human analysis:

```
                                  ┌─── make run ───► Headless CLI Loop (stdout / CI exit codes)
app.py (Marimo Reactive Script) ──┤
                                  └─── make edit ──► Reactive Browser UI (DAG inspection / visual exploration)
```

* **Headless CLI Mode (`make run`):**  
  Executes unbuffered directly in your shell. It streams progress metrics without allocating WebSockets or HTTP listeners, returning status code `0` on successful 100-pass invariant matching, and status code `1` on any detected byte divergence. Built specifically for CI runners and automated regression testing.
* **Interactive Reactive Mode (`make edit`):**  
  Spins up the local reactive web application (`http://localhost:2718`). Cells are organized as an Abstract Syntax Tree (AST) Directed Acyclic Graph (DAG), enabling real-time experimentation with prompt variations, generation lengths, and greedy boundaries while displaying formatted markdown outputs.

---

## Low-Resource Architecture & Memory Engineering

Standard LLM inference scripts frequently crash entry-level development machines (e.g., 8GB RAM laptops) due to aggressive multi-threading and duplicate memory allocations during weight deserialization. This harness is hardened against memory starvation through four architectural choices:

| Engineering Constraint | Implementation Mechanism | System Benefit |
| :--- | :--- | :--- |
| **Weight Footprint** | 4-bit Medium Quantization (`Q4_K_M`) | Download size capped at **~105 MB**; weights parse directly into memory without secondary tensor transformation passes. |
| **Peak Runtime RSS** | `low_cpu_mem_usage=True` + immediate `gc.collect()` | Caps active process memory consumption at **~250 MB**, completely eliminating Linux OOM (Out-Of-Memory) killer triggers. |
| **Thread Throttling** | `torch.set_num_threads(2)` clamped | Prevents CPU core saturation, keeping IDEs (e.g., VS Code), terminal multiplexers, and OS heartbeat processes fully responsive. |
| **Bandwidth Optimization** | Accelerated CDN mirror fallback via `hf-mirror.com` | Prevents Cloudflare network throttling, timeout drops, and connection hangs on unauthenticated Hugging Face endpoints. |

---

## DevX Automation: Makefile Primitives

All lifecycle operations are centralized inside the project `Makefile` to ensure clean, reproducible execution across heterogeneous local developer setups:

```bash
make <target>
```

| Target | Underlying Command | Technical Function |
| :--- | :--- | :--- |
| `make install` | `uv sync` | Resolves `uv.lock` and provisions an isolated virtual environment containing CPU wheels only. |
| `make download`| `curl -f -L -C - ...` | Retrieves the pre-quantized 105 MB GGUF binary with auto-resume support into the gitignored `models/` directory. |
| `make run` | `uv run python app.py` | Runs the full 100-pass headless verification harness with CLI progress updates and SHA-256 byte validation. |
| `make edit` | `uv run marimo edit app.py` | Launches the interactive reactive marimo server for visual in-browser inspection. |
| `make test` | `uv run pip list \| grep ...` | Audits the active environment to confirm complete absence of leaked `nvidia-*`, `cuda-*`, or `triton` libraries. |
| `make clean` | `rm -rf .venv uv.lock models/ ...` | Purges all local virtual environments, bytecode caches, temporary marimo states, and model binaries. |

---

## Dependency Isolation & Purity Audit

Standard PyPI distributions of PyTorch automatically pull down several gigabytes of compiled CUDA drivers (`nvidia-cusolver`, `nvidia-cudnn`, `triton`), creating massive environment bloat.

This repository enforces pure CPU-bound isolation via explicit wheel indexing in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "[https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

To audit your local environment for zero CUDA leakage:

```bash
make test
```

Expected output:

```text
[*] Checking for leaked GPU/CUDA dependencies...
[✓] Environment is 100% clean CPU-only.
```

---

## Repository Structure

```text
deterministic-llm-cpu/
├── .gitignore          # Strict exclusions for .venv, models/, *.gguf, and runtime caches
├── pyproject.toml      # Pinned CPU-only index and minimal dependencies (torch, transformers, marimo, gguf)
├── Makefile            # Uniform DevX interface (install, download, run, edit, test, clean)
├── README.md           # Theoretical grounding, empirical proof, guardrails, and setup guide
└── app.py              # Single-file dual runner: Marimo reactive dashboard & headless CLI evaluation engine
```

---

## Architectural Takeaways for AI Systems Engineering

Demonstrating that an autoregressive model produces deterministic byte outputs under fixed constraints yields a vital conclusion for production AI engineers:

1. **The Model is an Algebraic State Machine:** At inference time with temperature $T = 0$, a language model is not an unpredictable entity. It is a frozen, parameter-locked matrix transformation pipeline.
2. **Epistemic Fragility Persists:** Computational determinism does not imply reasoning truth. A model can deterministically repeat a hallucination indefinitely.
3. **The Necessity of the "Deterministic Shell":** Because language models operate via statistical inductive representation rather than symbolic verification, production architectures must never expose raw model outputs directly to mission-critical infrastructure. The role of the engineer is to enclose statistical models inside rigid, deterministic wrappers:
   * **Schema Conformance:** Enforcing strict Pydantic schemas on generated JSON.
   * **Syntax Analysis:** Passing generated SQL queries or scripts through static Abstract Syntax Tree (AST) parsers before database execution.
   * **Sandboxed Boundaries:** Isolating all generated operations under strict Least Privilege access rules.
   * **Automated Evals:** Running continuous CI regressions to measure semantic drift across model checkpoint updates.