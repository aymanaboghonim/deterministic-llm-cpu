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
        # Deterministic SLM Inference on CPU
        Bit-level verification of greedy autoregressive decoding using **marimo** and **Qwen2.5-0.5B-Instruct**.
        """
    )
    return


@app.cell
def __():
    import os
    import sys
    import hashlib
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Hardware & OS-level determinism
    os.environ["PYTHONHASHSEED"] = "0"
    torch.use_deterministic_algorithms(True)

    # 2. Concurrency clamping (prevents multi-core reduction tree divergence)
    torch.set_num_threads(4)
    torch.manual_seed(42)

    device = torch.device("cpu")
    dtype = torch.float32

    return AutoModelForCausalLM, AutoTokenizer, device, dtype, hashlib, os, sys, torch


@app.cell
def __(AutoModelForCausalLM, AutoTokenizer, device, dtype, mo):
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    mo.md(f"**Model Ready:** `{MODEL_ID}` loaded on `{device}` ({dtype}) with evaluation mode enabled.")
    return MODEL_ID, model, tokenizer


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
def __(input_tokens, model, tokenizer, torch, hashlib):
    NUM_ITERATIONS = 100
    token_hashes = []
    decoded_outputs = []

    with torch.inference_mode():
        for _ in range(NUM_ITERATIONS):
            # Pure greedy decoding: arg_max token selection over the vocabulary
            output_ids = model.generate(
                **input_tokens,
                max_new_tokens=30,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Isolate generated tokens from the prompt
            gen_tokens = output_ids[0, input_tokens.input_ids.shape[1]:]

            # Compute SHA-256 hash directly on the raw int64 byte buffer
            token_bytes = gen_tokens.numpy().tobytes()
            token_hash = hashlib.sha256(token_bytes).hexdigest()
            token_hashes.append(token_hash)

            # Decode text representation
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            decoded_outputs.append(text)

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
        view = mo.md(
            f"""
            ### Verification Status: PASSED (100% Deterministic)
            - **Evaluations:** `{NUM_ITERATIONS} / {NUM_ITERATIONS}` identical passes
            - **SHA-256 Digest:** `{baseline_hash}`
            - **Canonical Output:** *"{baseline_text}"*
            """
        )
    else:
        view = mo.md("### Verification Status: FAILED (Divergence Detected)")
        sys.exit(1)

    view
    return (view,)


if __name__ == "__main__":
    app.run()