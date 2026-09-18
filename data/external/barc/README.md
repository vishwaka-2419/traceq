# Third-party demand traces

These 64 files are the "real traces" of the BARC artifact that accompanies

> B. Ye, A. A. Khan and P. Liang, *When T-Depth Misleads: Predicting Fault-Tolerant Quantum Execution
> Slowdown under Magic-State Delivery Constraints*, arXiv:2604.11409.

Source: https://github.com/C2-Q/BARC, folder `barc_stage1/data/real_traces`, commit
`5c4421c82bcccc646ee7c79e76f983d76461d1ee`. The artifact is distributed under the Apache License 2.0
(copy in `LICENSE-APACHE-2.0.txt`). The traces were produced by its authors from Qiskit circuits
(ripple-carry and carry-lookahead adders, multipliers, modular multipliers, QFT and QAOA MaxCut)
transpiled to the basis {h, s, sdg, cx, t, tdg}; each entry is the number of T and T-dagger gates in
one layer of the circuit.

Changes made here: the step index was dropped and the demand column is stored as one unsigned byte
per step with xz compression (`<name>.u8.xz`). `traceq.external.load_trace` reads the files and
`scripts/fetch_external_traces.py` rebuilds them from the source repository. No demand value was changed.
