# Native llama.cpp TTT

The default runtime uses llama.cpp for inference and adaptation. It does not
start Python or convert HF checkpoints to GGUF after each update.

- A background training context uses the configured Zenzai device (4070 SUPER
  on this installation). It starts lazily when an example needs processing.
- Confirmed conversions retain the two-second debounce. The last confirmation
  for each reading in a batch wins.
- GPT-2 MLP down projections are adapted with the existing rank-update rule.
  Explicit word registrations fit the final projection and bias using cached
  activations and output-only gradients.
- Published weight versions are immutable. Training detaches shared projections
  before its next update. Inference adopts a finished version between conversion
  requests; it does not wait for the training pass or snapshot I/O.
- Only the mutable projections and final bias are written to atomic, fsynced
  `native-ttt-*.bin` snapshots. Snapshot identity includes the base GGUF path,
  size and modification time. The file includes the consumed queue cursor.
- `worker.lock` prevents simultaneous use of the legacy Python worker.

The existing GGUF and HF state remain usable as a rollback source. Native
snapshots are runtime state, not standalone GGUF releases. The vendored
AzooKeyKanaKanjiConverter adds the native API bridge to the bundled llama.cpp
fork. The Python script remains an explicit legacy tool.

## Validation tools

- `native_ttt_probe.cpp`: isolated CUDA1 model checks, word learning, immutable
  versions, snapshots, invalid inputs, numerical dumps and concurrent timing.
  Compile with the bundled llama/ggml headers and link the matching libraries.
  Arguments: backend directory, base F16 GGUF, empty output directory.
- `check_native_ttt_reference.py OUTPUT`: NumPy CPU reference for all 12 layers
  of zenz-v2.5-small. Run with one BLAS thread to avoid unnecessary CPU load.
- `native_ttt_e2e.py --model BASE.gguf`: an isolated server/protocol probe. It
  measures candidate conversion and input responses during word learning and
  normal TTT. Requires protoc, protobuf, and pynvml. It never uses the live
  server's socket or learning directory. `--library-dir` can select build output.

Quality and sequential retention are diagnostics, distinct from core API test
assertions. A successful rank update does not imply every target probability
improves or that subsequent updates preserve every previous greedy output.
