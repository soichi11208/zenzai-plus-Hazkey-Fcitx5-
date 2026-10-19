#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Check probe dumps against the GPT-2-small CPU rank-update reference."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
root = args.directory
ids = np.loadtxt(root / 'tokens.txt', dtype=np.int64)
embedding = np.fromfile(root / 'embedding.bin', np.float32).reshape(-1, 768)
next_values = embedding[ids[1:]]
results = []
for index in range(12):
    z = np.fromfile(root / f'z-{index}.bin', np.float32).reshape(len(ids), 3072)[:-1]
    before = np.fromfile(root / f'before-{index}.bin', np.float32).reshape(768, 3072)
    after = np.fromfile(root / f'after-{index}.bin', np.float32).reshape(768, 3072)
    expected = before + np.float32(1e-4) * (next_values.T @ z)
    error = float(np.max(np.abs(after - expected)))
    results.append({'layer': index, 'max_absolute_error': error})
    if error >= 2e-6:
        raise AssertionError((index, error))
report = {'passed': True, 'layers': results}
(root / 'reference-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
