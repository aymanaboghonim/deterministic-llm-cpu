import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # Deterministic SLM Inference on CPU (Local Pre-Quantized GGUF)
        Bit-level verification of greedy autoregressive decoding using **marimo** and **SmolLM2-135M-Instruct** (Q4_K_M GGUF).
        """
    )
    return


@app.cell
def __():
    import os
    import gc
    import sys
    import hashlib
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. CDN mirror to prevent download timeouts and throttling
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["PYTHONHASHSEED"] = "0"

    # 2. Hardware determinism
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 3. Restrict CPU thread concurrency to prevent CPU starvation
    torch.set_num_threads(2)
    torch.manual_seed(42)

    device = torch.device("cpu")
    dtype = torch.float32

    return AutoModelForCausalLM, AutoTokenizer, device, dtype, gc, hashlib, os, sys, torch


@app.cell
def __(AutoModelForCausalLM, AutoTokenizer, device, dtype, gc, mo, os):
    LOCAL_MODEL_DIR = "models"
    GGUF_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
    LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, GGUF_FILE)
    TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Pre-quantized binary not found at '{LOCAL_MODEL_PATH}'. "
            f"Execute 'make download' before running inference."
        )

    print(f"[*] Loading pre-quantized binary directly from disk: {LOCAL_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        gguf_file=GGUF_FILE,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)
    model.eval()

    gc.collect()
    print("[✓] Pre-quantized GGUF loaded into CPU memory from local storage.")

    ui_status = mo.md(f"**Model Ready:** `{GGUF_FILE}` loaded locally on `{device}`.")
    return (
        GGUF_FILE,
        LOCAL_MODEL_DIR,
        LOCAL_MODEL_PATH,
        TOKENIZER_ID,
        model,
        tokenizer,
        ui_status,
    )



@app.cell
def __(device, tokenizer):
    prompt = "Explain in exactly one sentence what causes a race condition in concurrent computing."
    messages = [
        {"role": "system", "content": "You are a precise technical assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_tokens = tokenizer(formatted_input, return_tensors="pt").to(device)
    return formatted_input, input_tokens, messages, prompt


@app.cell
def __(hashlib, input_tokens, model, sys, tokenizer, torch):
    NUM_ITERATIONS = 100
    token_hashes = []
    decoded_outputs = []

    print(f"[*] Starting {NUM_ITERATIONS} deterministic decoding iterations on CPU...")

    with torch.inference_mode():
        for i in range(1, NUM_ITERATIONS + 1):
            # Pure greedy decoding: arg_max token selection
            output_ids = model.generate(
                **input_tokens,
                max_new_tokens=25,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Extract generated tokens
            gen_tokens = output_ids[0, input_tokens.input_ids.shape[1]:]

            # SHA-256 digest on raw tensor bytes
            token_bytes = gen_tokens.numpy().tobytes()
            token_hash = hashlib.sha256(token_bytes).hexdigest()
            token_hashes.append(token_hash)

            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            decoded_outputs.append(text)

            # Zero-dependency terminal progress bar
            if i % 5 == 0 or i == 1 or i == NUM_ITERATIONS:
                progress = int((i / NUM_ITERATIONS) * 20)
                bar = "=" * progress + ("" if progress == 20 else ">") + " " * (20 - progress - (1 if progress < 20 else 0))
                percent = int((i / NUM_ITERATIONS) * 100)
                sys.stdout.write(f"\r    [{bar}] {i:3d}/{NUM_ITERATIONS} ({percent:3d}%) | Digest: {token_hash[:10]}...")
                sys.stdout.flush()

    print()  # Newline after loop

    baseline_hash = token_hashes[0]
    baseline_text = decoded_outputs[0]

    all_hashes_match = all(h == baseline_hash for h in token_hashes)
    all_texts_match = all(t == baseline_text for t in decoded_outputs)

    return (
        NUM_ITERATIONS,
        all_hashes_match,
        all_texts_match,
        baseline_hash,
        baseline_text,
        decoded_outputs,
        token_hashes,
    )


@app.cell
def __(
    NUM_ITERATIONS,
    all_hashes_match,
    all_texts_match,
    baseline_hash,
    baseline_text,
    mo,
    sys,
):
    if all_hashes_match and all_texts_match:
        # Mode 1: Terminal stdout for CLI and CI
        print("\n" + "=" * 65)
        print("VERIFICATION STATUS: PASSED (100% Deterministic)")
        print(f"Iterations Checked : {NUM_ITERATIONS} / {NUM_ITERATIONS} identical passes")
        print(f"SHA-256 Digest     : {baseline_hash}")
        print(f"Canonical Output   : \"{baseline_text}\"")
        print("=" * 65 + "\n")

        # Mode 2: Reactive Markdown UI for marimo browser view
        view = mo.md(
            f"""
            ### Verification Status: PASSED (100% Deterministic)
            - **Evaluations:** `{NUM_ITERATIONS} / {NUM_ITERATIONS}` identical passes
            - **SHA-256 Digest:** `{baseline_hash}`
            - **Canonical Output:** *"{baseline_text}"*
            """
        )
    else:
        print("\n[!] VERIFICATION FAILED: Output variance detected.\n", file=sys.stderr)
        view = mo.md("### Verification Status: FAILED (Divergence Detected)")
        sys.exit(1)

    return (view,)


if __name__ == "__main__":
    app.run()