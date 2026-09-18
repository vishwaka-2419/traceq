#!/usr/bin/env python3
"""Fetch the public BARC traces of Ye, Khan and Liang and store them compactly.

Source : https://github.com/C2-Q/BARC (Apache License 2.0), folder barc_stage1/data/real_traces
Pinned : commit 5c4421c82bcccc646ee7c79e76f983d76461d1ee
Output : data/external/barc/<name>.u8.xz (one unsigned byte per step, xz-compressed)

The repository already contains the converted traces. This script documents their
provenance and lets anyone rebuild them. It needs git and network access.
"""
from __future__ import annotations
import argparse, subprocess, tempfile
from pathlib import Path
import numpy as np
from traceq.external import save_compact

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/C2-Q/BARC"
COMMIT = "5c4421c82bcccc646ee7c79e76f983d76461d1ee"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="existing clone of the BARC repository")
    a = ap.parse_args()
    dest = ROOT / "data/external/barc"; dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = a.source
        if src is None:
            src = Path(tmp) / "BARC"
            subprocess.run(["git", "clone", REPO, str(src)], check=True)
            subprocess.run(["git", "-C", str(src), "checkout", COMMIT], check=True)
        files = sorted((src / "barc_stage1/data/real_traces").glob("*.csv"))
        for f in files:
            data = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
            if not np.array_equal(data[:, 0], np.arange(len(data))):
                raise SystemExit(f"{f.name}: unexpected step index")
            save_compact(data[:, 1], dest / (f.stem + ".u8.xz"))
        (dest / "LICENSE-APACHE-2.0.txt").write_bytes((src / "LICENSE").read_bytes())
    print(f"stored {len(files)} traces in {dest}")


if __name__ == "__main__":
    main()
