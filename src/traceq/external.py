"""Readers for demand traces stored outside the package formats.

``.csv`` and ``.csv.gz`` files need a header and the demand in the last column.
``.u8.xz`` files hold one unsigned byte per scheduled step, xz-compressed; the step
index is implicit. This compact form is used for the third-party traces in
``data/external``.
"""
from __future__ import annotations
import gzip, lzma
from pathlib import Path
import numpy as np

__all__ = ["load_trace", "save_compact"]


def load_trace(path):
    path = Path(path)
    name = path.name
    if name.endswith(".u8.xz"):
        with lzma.open(path, "rb") as f:
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    return data[:, -1].copy()


def save_compact(demand, path):
    m = np.asarray(demand)
    if m.ndim != 1 or m.min() < 0 or m.max() > 255:
        raise ValueError("compact format needs a one-dimensional trace with entries in 0..255")
    with lzma.open(path, "wb", preset=9 | lzma.PRESET_EXTREME) as f:
        f.write(m.astype(np.uint8).tobytes())
