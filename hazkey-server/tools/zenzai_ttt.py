#!/usr/bin/env python3
"""Local In-Place TTT worker for Hazkey and zenz-v2.5-small.

Each confirmed IME conversion is one causal document. For every GPT-2 block,
the final MLP projection is updated with the paper's LM-aligned rule
    W_down += eta * V_next.T @ Z
where Z is the post-GELU MLP activation and V_next is the next-token embedding.
The updated full checkpoint is exported as a new GGUF generation. Hazkey picks
up that generation on the next conversion by changing the model URL.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Standalone workers use the same runtime GPU as the Hazkey adapter.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HAZKEY_TTT_CUDA_DEVICE", "1"))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open


def katakana(text: str) -> str:
    return "".join(chr(ord(char) + 0x60) if "ぁ" <= char <= "ゖ" else char for char in text)


def prompt_for(example: dict[str, str], tokenizer: AutoTokenizer) -> str:
    prompt = "\uEE00" + katakana(example["reading"])
    if example.get("context"):
        prompt += "\uEE02" + example["context"][-40:]
    prompt += "\uEE01" + example["output"]
    return prompt + (tokenizer.eos_token or "")


def apply_ttt_example(model, tokenizer, example: dict[str, str], device: torch.device, lr: float) -> None:
    token_ids = tokenizer(prompt_for(example, tokenizer), return_tensors="pt").input_ids.to(device)
    if token_ids.shape[1] < 2:
        return

    activations: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_index, block in enumerate(model.transformer.h):
        def capture(_module, inputs, index=layer_index):
            activations[index] = inputs[0].detach()

        hooks.append(block.mlp.c_proj.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            model(input_ids=token_ids, use_cache=False)
            next_embeddings = model.transformer.wte(token_ids[:, 1:]).detach()
        # One confirmed conversion is a causal update chunk. The next example
        # sees the newly adapted projections, as in the sequential TTT rule.
        with torch.no_grad():
            for layer_index, block in enumerate(model.transformer.h):
                z = activations[layer_index][:, :-1, :].reshape(-1, block.mlp.c_proj.weight.shape[0])
                v = next_embeddings.reshape(-1, next_embeddings.shape[-1])
                update = z.transpose(0, 1) @ v
                block.mlp.c_proj.weight.add_(update, alpha=lr)
    finally:
        for hook in hooks:
            hook.remove()


def read_examples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # ignore a line while Hazkey is still appending it
                if row.get("reading") and row.get("output"):
                    result.append(row)
    return result


def latest_examples(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the last confirmation for each reading, in confirmation order."""
    seen = set()
    selected = []
    for example in reversed(examples):
        reading = katakana(example["reading"])
        if reading not in seen:
            seen.add(reading)
            selected.append(example)
    return list(reversed(selected))


def resolve_source_model(data_dir: Path, base_model: str) -> Path:
    """Use an installed local base when pinned, including after history reset."""
    manifest = data_dir / "base-model.json"
    if manifest.is_file():
        local = Path(json.loads(manifest.read_text(encoding="utf-8"))["local_model"]).expanduser()
        source_model = local if local.is_absolute() else data_dir / local
    elif Path(base_model).expanduser().is_dir():
        source_model = Path(base_model).expanduser()
    else:
        source_model = data_dir / "source-model"
        snapshot_download(repo_id=base_model, local_dir=source_model)
        return source_model
    source_model = source_model.resolve()
    for filename in ("config.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
        if not (source_model / filename).is_file():
            raise FileNotFoundError(f"Installed TTT base is missing {filename}: {source_model}")
    if not (source_model / "model.safetensors").is_file() and not (source_model / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"Installed TTT base has no model weights: {source_model}")
    return source_model


def validate_weights(items):
    for name, tensor in items:
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Non-finite model weight: {name}")
        if tensor.ndim >= 2 and not bool(torch.count_nonzero(tensor)):
            raise RuntimeError(f"All-zero model weight: {name}")


def save_generation(model, args, data_dir: Path, example_count: int) -> None:
    validate_weights(model.named_parameters())
    work_dir = data_dir / "hf-checkpoint.next"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    model.save_pretrained(work_dir, safe_serialization=True)
    weights = work_dir / "model.safetensors"
    with weights.open("rb") as stream:
        os.fsync(stream.fileno())
    with safe_open(weights, framework="pt", device="cpu") as saved:
        validate_weights((name, saved.get_tensor(name)) for name in saved.keys())
    # save_pretrained() on a model only saves weights and model configuration.
    # The converter also needs the original vocabulary and tokenizer settings;
    # otherwise it can fall back to the stock GPT-2 tokenizer.
    source_model = getattr(args, "source_model", data_dir / "source-model")
    for filename in (
        "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json",
    ):
        source = source_model / filename
        if source.is_file():
            shutil.copy2(source, work_dir / filename)

    generations = data_dir / "models"
    generations.mkdir(parents=True, exist_ok=True)
    generation_name = f"zenzai-ttt-{example_count:08d}{getattr(args, 'generation_tag', '')}.gguf"
    destination = generations / generation_name
    with tempfile.TemporaryDirectory(prefix="zenzai-gguf-", dir=generations) as temporary:
        temporary_output = Path(temporary) / generation_name
        converter = args.converter or os.environ.get("HAZKEY_ZENZAI_TTT_CONVERTER")
        if not converter or not Path(converter).is_file():
            raise RuntimeError("llama.cpp convert_hf_to_gguf.py is unavailable")
        command = [sys.executable, converter, str(work_dir), "--outfile", str(temporary_output), "--outtype", "f16"]
        subprocess.run(command, check=True)
        os.replace(temporary_output, destination)

    checkpoint = data_dir / "hf-checkpoint"
    old_checkpoint = data_dir / "hf-checkpoint.old"
    if old_checkpoint.exists():
        shutil.rmtree(old_checkpoint)
    if checkpoint.exists():
        os.replace(checkpoint, old_checkpoint)
    os.replace(work_dir, checkpoint)
    if old_checkpoint.exists():
        shutil.rmtree(old_checkpoint)

    progress_tmp = data_dir / "progress.json.tmp"
    progress_tmp.write_text(json.dumps({"processed": example_count}), encoding="utf-8")
    os.replace(progress_tmp, data_dir / "progress.json")
    pointer_tmp = data_dir / "active-model.txt.tmp"
    pointer_tmp.write_text(str(destination) + "\n", encoding="utf-8")
    os.replace(pointer_tmp, data_dir / "active-model.txt")
    for old_generation in generations.glob("zenzai-ttt-*.gguf"):
        if old_generation != destination:
            old_generation.unlink(missing_ok=True)


def write_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def register_word(data_dir, row):
    """Persist lattice entries independently of model training success."""
    path = data_dir / "registered-words.json"
    words = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    word = {"reading": katakana(row["reading"].strip()), "output": row["output"].strip()}
    if not word["reading"] or not word["output"]:
        raise ValueError("読みと単語が必要です")
    if word not in words:
        words.append(word)
        write_status(path, words)


def word_tokens(tokenizer, row, device):
    prefix = "\uEE00" + katakana(row["reading"]) + "\uEE01"
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    expected = tokenizer(row["output"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    target = torch.tensor(expected, device=device)
    return torch.cat([prefix_ids, target.unsqueeze(0)], dim=1), target


def predicts_word(model, tokenizer, row, device):
    ids, target = word_tokens(tokenizer, row, device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0, -len(target)-1:-1]
        return bool(torch.equal(logits.argmax(-1), target))


def learn_word(model, tokenizer, row, device, limit, report):
    """Fit only the final MLP projection on cached features; no transformer backprop."""
    started = time.perf_counter()
    ids, target = word_tokens(tokenizer, row, device)
    projection = model.transformer.h[-1].mlp.c_proj
    cache = {}
    def capture_z(_module, inputs):
        cache["z"] = inputs[0].detach()
    def capture_hidden(_module, _inputs, output):
        cache["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()
    hooks = [projection.register_forward_pre_hook(capture_z),
             model.transformer.h[-1].register_forward_hook(capture_hidden)]
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    z = cache["z"][:, -len(target)-1:-1].detach()
    hidden = cache["hidden"][:, -len(target)-1:-1].detach()
    with torch.no_grad():
        residual = hidden - projection(z)
    original = {name: value.detach().clone() for name, value in projection.state_dict().items()}
    original_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    steps = 0
    matched = False
    try:
        for parameter, _ in original_flags:
            parameter.requires_grad_(False)
        for parameter in projection.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(projection.parameters(), lr=0.002, weight_decay=0)
        for step in range(limit + 1):
            logits = model.lm_head(model.transformer.ln_f(residual + projection(z)))[0]
            matched = bool(torch.equal(logits.argmax(-1), target))
            if matched or step == limit or time.perf_counter() - started > 0.5:
                break
            loss = torch.nn.functional.cross_entropy(logits, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite word-learning loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            optimizer.step()
            steps += 1
            report(steps)
        if matched:
            matched = predicts_word(model, tokenizer, row, device)
        if not matched:
            projection.load_state_dict(original)
    except Exception:
        projection.load_state_dict(original)
        raise
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, flag in original_flags:
            parameter.requires_grad_(flag)
    return steps, matched, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="watch the Hazkey append-only example queue")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--base-model", default=os.environ.get("HAZKEY_ZENZAI_HF_MODEL", "Miwa-Keita/zenz-v2.5-small"))
    parser.add_argument("--base-gguf", type=Path)
    parser.add_argument("--converter", default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--debounce-seconds", type=float, default=2.0)
    args = parser.parse_args()

    os.umask(0o077)
    data_dir = args.data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    worker_lock = (data_dir / "worker.lock").open("a+")
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    queue = data_dir / "examples.jsonl"
    checkpoint = data_dir / "hf-checkpoint"
    progress_path = data_dir / "progress.json"
    pointer = data_dir / "active-model.txt"
    converter = args.converter or os.environ.get("HAZKEY_ZENZAI_TTT_CONVERTER")
    if not converter or not Path(converter).is_file():
        raise RuntimeError("llama.cpp convert_hf_to_gguf.py is unavailable")

    source_model = resolve_source_model(data_dir, args.base_model)
    args.source_model = source_model

    base_gguf = args.base_gguf or data_dir / "base.gguf"
    if not base_gguf.is_file():
        base_gguf.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="zenzai-base-", dir=data_dir) as temporary:
            temporary_base = Path(temporary) / "base.gguf"
            subprocess.run(
                [sys.executable, converter, str(source_model), "--outfile", str(temporary_base), "--outtype", "f16"],
                check=True,
            )
            os.replace(temporary_base, base_gguf)
    if not pointer.exists():
        pointer.write_text(str(base_gguf) + "\n", encoding="utf-8")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"In-Place TTT device: {device}; physical GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}", flush=True)
    model = None
    tokenizer = None

    requests = data_dir / "word-requests"
    requests.mkdir(exist_ok=True, mode=0o700)
    # Migrate earlier registrations, including ones whose TTT did not converge.
    for old_request in sorted(requests.glob("*.done")):
        try:
            register_word(data_dir, json.loads(old_request.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError) as error:
            print(f"Skipping invalid old registration {old_request.name}: {error}", flush=True)

    while True:
        reset_flag = data_dir / "reset"
        if reset_flag.exists():
            reset_flag.unlink(missing_ok=True)
            for path in (checkpoint, data_dir / "hf-checkpoint.next", data_dir / "models"):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            queue.write_text("", encoding="utf-8")
            progress_path.unlink(missing_ok=True)
            pointer_tmp = data_dir / "active-model.txt.tmp"
            pointer_tmp.write_text(str(base_gguf) + "\n", encoding="utf-8")
            os.replace(pointer_tmp, pointer)
            model = None
            tokenizer = None

        examples = read_examples(queue)
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"processed": 0}
        processed = int(progress["processed"])
        pending = sorted(requests.glob("*.json"))
        if pending:
            request_path = pending[0]
            status_path = request_path.with_suffix(".status")
            try:
                row = json.loads(request_path.read_text(encoding="utf-8"))
                if not isinstance(row.get("reading"), str) or not isinstance(row.get("output"), str):
                    raise ValueError("読みと単語が必要です")
                if not row["reading"].strip() or not row["output"].strip() or max(len(row["reading"]), len(row["output"])) > 128:
                    raise ValueError("読みと単語は1〜128文字で指定してください")
                register_word(data_dir, row)
                limit = min(128, max(1, int(row.get("max_steps", 32))))
                write_status(status_path, {"state": "training", "steps": 0, "dictionary_registered": True})
                if model is None:
                    tokenizer = AutoTokenizer.from_pretrained(source_model)
                    model = AutoModelForCausalLM.from_pretrained(
                        checkpoint if checkpoint.exists() else source_model,
                        torch_dtype=torch.float32,
                    ).to(device)
                    model.eval()
                    model.config.use_cache = False
                validate_weights(model.named_parameters())
                def report(steps):
                    write_status(status_path, {"state": "training", "steps": steps, "dictionary_registered": True})
                steps, matched, elapsed = learn_word(model, tokenizer, row, device, limit, report)
                export_started = time.perf_counter()
                if matched and steps:
                    args.generation_tag = "-word-" + str(time.time_ns())
                    save_generation(model, args, data_dir, processed)
                    args.generation_tag = ""
                write_status(status_path, {"state": "done", "steps": steps, "matched": matched,
                    "dictionary_registered": True, "method": "cached-final-mlp-gradient",
                    "training_seconds": elapsed, "export_seconds": time.perf_counter() - export_started,
                    "model_updated": bool(matched and steps)})
                print(f"Word learning: matched={matched}, steps={steps}, train={elapsed:.3f}s", flush=True)
            except Exception as error:
                write_status(status_path, {"state": "error", "message": str(error)})
                # Reload the last published model if training/export failed.
                model = None
                tokenizer = None
                args.generation_tag = ""
            request_path.rename(request_path.with_suffix(".done"))
            continue
        if len(examples) > processed:
            observed_mtime = queue.stat().st_mtime
            time.sleep(args.debounce_seconds)
            if queue.stat().st_mtime != observed_mtime:
                continue
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(source_model)
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint if checkpoint.exists() else source_model,
                    torch_dtype=torch.float32,
                ).to(device)
                model.eval()
                model.config.use_cache = False
            validate_weights(model.named_parameters())
            update_started = time.perf_counter()
            batch = examples[processed:]
            updates = latest_examples(batch)
            for row in updates:
                apply_ttt_example(model, tokenizer, row, device, args.learning_rate)
            # Advance past every queued record, including superseded readings.
            processed += len(batch)
            save_generation(model, args, data_dir, processed)
            print(f"In-Place TTT updated through {processed} confirmed conversions; elapsed={time.perf_counter()-update_started:.3f}s, device={device}", flush=True)
        elif not args.watch:
            return
        else:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
