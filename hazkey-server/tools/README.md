# Zenzai In-Place TTT

Hazkey's server starts `zenzai_ttt.py` as a local background worker. It fetches
`Miwa-Keita/zenz-v2.5-small` into Hazkey's private data directory, converts the
base checkpoint to GGUF, and then waits for confirmed IME conversions. Every
confirmed reading/output pair is appended locally. Within each update batch,
only the last confirmation for each reading is used as a causal TTT chunk;
hiragana and katakana readings are treated as equivalent. Retained examples
are applied in confirmation order. The worker applies the In-Place TTT next-token-embedding update to the
final MLP projection in each GPT-2 block, exports a new GGUF generation, and
atomically switches the active model pointer.

The default paths target a source checkout. `HAZKEY_ZENZAI_TTT_SCRIPT`,
`HAZKEY_ZENZAI_TTT_CONVERTER`, `HAZKEY_ZENZAI_HF_MODEL`, `HAZKEY_PYTHON`, and
`HAZKEY_TTT_CUDA_DEVICE` can override them. The TTT worker needs Python packages
`torch`, `transformers`, `huggingface_hub`, `safetensors`, `numpy`, and the
`gguf` package shipped with llama.cpp. It uses visible CUDA device 0 for updates
by default; on the specified machine this is the RTX 3090. Hazkey selects the
available inference backend whose description names the RTX 4070 SUPER when the
profile uses the default CPU setting. An explicit backend selection takes priority.

## CUDA inference build

The local installation uses CUDA 12.8 with GCC 13 as its host compiler. From the
repository root, enable the matching llama.cpp CUDA backend with:

```sh
cmake -S . -B /tmp/hazkey-install-build \
  -DGGML_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  '-DCMAKE_CUDA_ARCHITECTURES=86;89'
cmake --build /tmp/hazkey-install-build --parallel 6
cmake --install /tmp/hazkey-install-build
```

This command reuses the existing build configured for `~/.local`. In this
environment, the Hazkey profile selects `CUDA1` (RTX 4070 SUPER) for inference;
the Python TTT worker uses CUDA device 0 (RTX 3090) independently. Device indices
must be checked again if the GPU enumeration order changes.

`clearAllHistory` also resets the TTT queue and personal checkpoint while
preserving the downloaded base model. The model files and confirmed conversion
history stay under Hazkey's user data directory and are not uploaded.
