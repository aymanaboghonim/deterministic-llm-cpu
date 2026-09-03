# Deterministic LLM Inference on CPU

A minimal, reproducible harness to prove **computational bit-level determinism** in autoregressive Small Language Models (SLMs) running locally on CPU architectures without CUDA dependencies.

---

## Scientific Scope & Guardrails

### 1. The Core Thesis: Computational Determinism vs. Epistemic Validity

This experiment isolates and proves **Computational Determinism**; it does not validate **Epistemic Truth**.

* **Computational Determinism (Verified):** Given an identical model checkpoint, invariant runtime dependencies, fixed CPU threading, and greedy decoding ($T = 0$, `do_sample=False`), an autoregressive language model behaves as a deterministic, closed algebraic function:
  $$f(X; \theta) = \mathbf{z} \implies x_t = \arg\max_{i \in \mathcal{V}} (z_i)$$
  Under these exact hardware constraints, every forward pass generates an identical sequence of token IDs down to the exact byte-level representation across $N = 100$ runs.
* **Epistemic Validity & Statistical Nature (Explicitly Not Solved):** The model's weights ($\theta$) remain lossy statistical approximations derived from pretraining corpora. If the greedy decoding path yields a hallucinated claim or a logical fallacy, the model will reproduce that exact error with 100% certainty across every iteration. Computational determinism guarantees **reproducibility of the output**, never **correctness of the statement**. Reliable production systems still require deterministic runtime shells (e.g., AST parsers, schema validators, sandboxed code execution).

---

### 2. Experimental Boundaries & Invalidation Conditions

The claims of this repository are valid **strictly** under the following execution boundary conditions:

| Parameter | Required Setting | Why It Breaks Otherwise |
| :--- | :--- | :--- |
| **Decoding Strategy** | `do_sample=False`, `num_beams=1` | Any stochastic sampling ($T > 0$, Top-$p$, Top-$k$) introduces pseudo-random number generator (PRNG) divergence. |
| **Concurrency & Batching** | Batch Size $B = 1$, Isolated Process | Dynamic batching and parallel user requests alter tensor memory offsets, introducing non-deterministic scheduling. |
| **Threading Strategy** | `torch.set_num_threads(N)` clamped | Uncontrolled multi-core reduction trees allow race conditions in floating-point addition order. |
| **Model Mode** | `model.eval()`, `torch.inference_mode()` | Evaluation mode disables stochastic layers such as Dropout. |
| **Hardware Target** | Single, identical CPU architecture | Cross-architecture executions (e.g., Intel AVX-512 vs. Apple Silicon NEON) alter Fused Multiply-Add (FMA) rounding order. |

---

### 3. Explicit Disclaimers & Non-Goals

1. **No Cross-Hardware Hash Guarantee:** The resulting SHA-256 digest is invariant **only on the specific physical CPU and compiler toolchain** running the code. A run on an AMD EPYC processor may produce a different token digest than an Intel Core or Apple M-series processor if intermediate floating-point logits differ by $10^{-6}$ near greedy boundary thresholds.
2. **No Claim of General Model Reliability:** This repository does not claim that small language models (SLMs) or generative models are inherently reliable or "non-stochastic" in business environments. Real-world systems run with asynchronous concurrency, GPU kernel optimizations, and floating-point shortcuts that intentionally sacrifice bit-level determinism for throughput.