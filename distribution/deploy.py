#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Install a staged CMake tree, preserving replaced files for uninstall."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

mode, prefix = sys.argv[1:3]
prefix = Path(prefix)
if not prefix.is_absolute() or prefix == Path('/') or '..' in prefix.parts:
    raise SystemExit('Invalid prefix')
state = prefix / 'share/zinai-plus-Hazkey-Fcitx5-install-state'
manifest = state / 'manifest.json'

def digest(p):
    if p.is_symlink(): return 'link:' + os.readlink(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink(): target.unlink()
    if source.is_symlink(): target.symlink_to(os.readlink(source))
    else: shutil.copy2(source, target)

if mode == 'install':
    if state.exists():
        raise SystemExit('An installation backup already exists. Uninstall it before reinstalling.')
    stage = Path(sys.argv[3])
    tree = stage / str(prefix).lstrip('/')
    # Reject files outside PREFIX; those would be missed by deployment/uninstall.
    for p in stage.rglob('*'):
        if (p.is_file() or p.is_symlink()) and not p.is_relative_to(tree):
            raise SystemExit(f'Staged file outside PREFIX: {p}')
    files = sorted(p for p in tree.rglob('*') if p.is_file() or p.is_symlink())
    if not files: raise SystemExit('Empty staged installation')
    entries = []
    state.mkdir(parents=True)
    try:
        for source in files:
            rel = source.relative_to(tree)
            target = prefix / rel
            saved = target.exists() or target.is_symlink()
            if saved: copy(target, state / 'backup' / rel)
            entry = {'path': str(rel), 'saved': saved, 'hash': digest(source)}
            entries.append(entry)
            # Persist the recovery manifest before replacing each file.
            manifest.write_text(json.dumps(entries, indent=2))
            copy(source, target)
    except BaseException:
        print(f'Installation interrupted. Recovery data: {state}', file=sys.stderr)
        raise
    print(f'Installed {len(entries)} files; replaced files saved in {state}')
elif mode == 'uninstall':
    entries = json.loads(manifest.read_text())
    remaining = []
    for entry in reversed(entries):
        rel = Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts: raise SystemExit('Invalid manifest path')
        target = prefix / rel
        if target.exists() or target.is_symlink():
            if digest(target) != entry['hash']:
                print(f'Kept modified file: {target}')
                remaining.append(entry)
                continue
            target.unlink()
        if entry['saved']: copy(state / 'backup' / rel, target)
    if remaining:
        manifest.write_text(json.dumps(list(reversed(remaining)), indent=2))
        print(f'Modified files and their backups retained in {state}')
    else:
        shutil.rmtree(state)
        print('Uninstalled; replaced files restored. User settings/history/models retained.')
else:
    raise SystemExit('Expected install or uninstall')
