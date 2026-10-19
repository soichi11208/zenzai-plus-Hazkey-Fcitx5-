#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Export an atomically saved native TTT snapshot as a standalone GGUF (CPU only)."""
import argparse
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile

# Works both from the source tree and installed libexec/hazkey directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'llama.cpp/gguf-py'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'llama.cpp/gguf-py'))


def main():
    import numpy as np
    import gguf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('XDG_DATA_HOME', str(Path.home()/'.local/share'))) / 'hazkey/zenzai-ttt')
    parser.add_argument('--model', type=Path, help='Base GGUF used by the snapshot')
    parser.add_argument('--fallback-model', type=Path, help='Model selected by Hazkey settings when no active override exists')
    parser.add_argument('--snapshot', type=Path, help='Explicit snapshot; must belong to --model')
    args = parser.parse_args()
    model = args.model
    if model is None:
        active = args.root / 'active-model.txt'
        active_path = Path(active.read_text().strip()) if active.exists() and active.read_text().strip() else None
        candidates = [active_path, args.fallback_model, args.root / 'base.gguf',
                      Path(os.environ['HAZKEY_ZENZAI_MODEL']) if os.environ.get('HAZKEY_ZENZAI_MODEL') else None,
                      args.root.parent / 'zenzai/zenzai.gguf']
        model = next((p for p in candidates if p is not None and p.is_file()), None)
        if model is None: raise ValueError('No base GGUF found; specify --model')
    model = model.resolve(strict=True)
    if (args.root / 'reset').exists():
        raise ValueError('TTT reset is pending; retry after the reset has completed')
    snapshot = args.snapshot
    if snapshot is None:
        st = model.stat()
        identity = f'{model}:{st.st_size}:{st.st_mtime}'
        h = 14695981039346656037
        for b in identity.encode(): h = ((h ^ b) * 1099511628211) & ((1 << 64)-1)
        snapshot = args.root / f'native-ttt-{h:x}.bin'
        if not snapshot.exists():
            raise ValueError('No saved TTT state for the active model. If using a custom model, specify --model and --snapshot explicitly.')
    output = args.output.expanduser().absolute()
    if output.exists(): raise ValueError(f'Output already exists: {output}')
    if not output.parent.is_dir(): raise ValueError('Output directory does not exist')
    # Open once: rename-based saves cannot mix generations in this byte buffer.
    raw = snapshot.read_bytes()
    if len(raw) < 32: raise ValueError('Incomplete snapshot')
    magic, cursor, layers, intermediate, hidden, reserved = struct.unpack('<8sQIIII', raw[:32])
    if magic != b'ZTTT0001' or reserved or not (0 < layers <= 1024 and intermediate > 0 and hidden > 0):
        raise ValueError('Invalid snapshot header')
    expected = 32 + 4*(layers*intermediate*hidden + hidden)
    if len(raw) != expected: raise ValueError('Snapshot length mismatch')
    values = np.frombuffer(raw, dtype='<f4', offset=32)
    if not np.isfinite(values).all(): raise ValueError('Non-finite snapshot')
    reader = gguf.GGUFReader(str(model))
    if reader.fields['general.architecture'].contents() != 'gpt2': raise ValueError('Only GPT2 Zenzai models are supported')
    replacements = {}
    offset = 0
    for i in range(layers):
        count = intermediate * hidden
        weight = values[offset:offset+count].reshape(hidden, intermediate)
        if not np.any(weight): raise ValueError('Zero snapshot weight')
        replacements[f'blk.{i}.ffn_down.weight'] = weight
        offset += count
    replacements[f'blk.{layers-1}.ffn_down.bias'] = values[offset:]
    tensors = {t.name: t for t in reader.tensors}
    for name, data in replacements.items():
        if name not in tensors or tuple(tensors[name].shape) != tuple(reversed(data.shape)):
            raise ValueError(f'Snapshot/model shape mismatch: {name}')
    if int(reader.fields['gpt2.block_count'].contents()) != layers:
        raise ValueError('Snapshot/model layer count mismatch')
    fd, temporary = tempfile.mkstemp(prefix='.ttt-export-', suffix='.gguf', dir=output.parent)
    os.close(fd)
    os.chmod(temporary, 0o600)
    writer = None
    try:
        writer = gguf.GGUFWriter(temporary, 'gpt2')
        for name, field in reader.fields.items():
            if name.startswith(('GGUF.', 'zinai_plus.ttt.')) or name in ('general.architecture', 'general.file_type'): continue
            writer.add_key_value(name, field.contents(), field.types[0])
        # Mixed types: preserve frozen tensor formats, keep learned tensors in F32.
        writer.add_uint32('general.file_type', 1024)
        writer.add_string('zinai_plus.ttt.snapshot_sha256', hashlib.sha256(raw).hexdigest())
        writer.add_uint64('zinai_plus.ttt.processed', cursor)
        for t in reader.tensors:
            if t.name in replacements: writer.add_tensor(t.name, replacements[t.name])
            else: writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        with open(temporary, 'rb') as stream: os.fsync(stream.fileno())
        # Do not overwrite an existing destination, even if created during export.
        os.link(temporary, output)
        os.unlink(temporary)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if writer is not None: writer.close()
        Path(temporary).unlink(missing_ok=True)
    print(f'Exported: {output}\nSaved TTT cursor: {cursor}\nStandalone GGUF; input history is not included.')

if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(f'Export failed: {error}', file=sys.stderr)
        sys.exit(1)
