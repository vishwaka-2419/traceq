# traceq

Code, data and figure scripts for the paper

> A. Vishwakarma, *Beyond Mean Factory Rate: Trace-Aware Provisioning for Magic-State Cultivation* (2026).

The paper studies a finite-buffer queue in which post-selected magic-state producers feed an ordered,
atomic demand trace. A rate gives a point heuristic for the runtime. The complete demand histogram
confines the runtime to an interval whose ends differ by a factor below two, and no better. The
ordered trace fixes the runtime exactly when service is deterministic, through an explicit max-plus
formula that one pass evaluates. The repository measures how far that formula carries when service is
random, when rotations advance asynchronously, when the synthesis-chain length and the service law
change, and on 64 independently produced traces.

The simulators, the closed forms, the Hubbard demand traces, every Monte Carlo realisation, all tables and all figures can be regenerated from repository.

## Contents

| Path | Purpose |
|---|---|
| `src/traceq/model.py` | Explicit blocking-after-service queue with atomic, ordered demands |
| `src/traceq/theory.py` | Order-sensitive closed forms for the deterministic limit of the queue |
| `src/traceq/asynchronous.py` | Lane-based consumer: rotations advance independently instead of in lockstep |
| `src/traceq/policies.py` | Integer-clock reference for alternative holding and injection policies |
| `src/traceq/analysis.py` | Order-sensitive fluid diagnostic and finite-sample reporting |
| `src/traceq/workloads.py` | Auditable Pauli-rotation demand proxies for the shifted Hubbard Hamiltonian |
| `src/traceq/protocols.py` | Optional stage-resolved supply adapter; parameters require physical calibration |
| `src/traceq/external.py` | Readers for demand traces stored outside the package formats |
| `src/traceq/cli.py` | Run an externally supplied demand trace: python -m traceq.cli --help |
| `tests/` | 276 tests: simulator cross-checks, a clock-tick oracle, invariants, the theorems of the paper, the asynchronous simulator |
| `scripts/run_study.py` | The main Monte Carlo study (248 batches, 86,280 realisations, about 7 minutes on one core) |
| `scripts/run_theory.py` | Closed-form analysis of every stored batch plus 22 small new batches (about 20 s) |
| `scripts/run_robustness.py` | Asynchronous progress, synthesis-chain length, staged service laws, external traces (about 5 minutes; `--part` runs one at a time) |
| `scripts/fetch_external_traces.py` | Rebuilds `data/external/barc/` from the public BARC repository at a pinned commit |
| `scripts/build_assets.py` | Tables and numeric macros of the paper, written to `generated/` |
| `scripts/make_figures.py` | All figures of the paper and supplement, written to `figures/` |
| `scripts/verify_artifacts.py` | Recomputes every summary statistic of the main study from the realisation-level data |
| `scripts/audit_replay.py` | Replays stored realisations through both simulators |
| `scripts/verify_full_reproduction.py` | Compares a fresh full rerun with the stored study |
| `data/traces/` | Hubbard demand traces (CSV) and full workload descriptions (JSON) |
| `data/external/barc/` | 64 third-party demand traces (Apache 2.0, see the README in that folder) |
| `data/results/` | `study.json`, `theory.json`, `robustness.json`, realisation-level CSV files, selected designs |
| `figures/` | Figures as PDF and PNG, numbered as in the paper |

## Install

Python 3.10 or newer.

```
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

`requirements-tested.txt` lists the package versions that produced the stored results.

## Quick start

```python
from traceq import (Config, simulate, deterministic_runtime, fluid_runtime,
                    histogram_envelope, cluster_family)

# Two traces with the same histogram (Theorem 1 of the paper): width 4, clusters of 2, 3 repeats
spread, clustered = cluster_family(width=4, cluster=2, repeats=3)

# One producer, buffer 4, unit step time, unit deterministic latency, full initial buffer
cfg = Config(producers=1, buffer=4, step_time=1, attempt_time=1, acceptance=1.0,
             prefill=True, law="deterministic")
print(simulate(spread, cfg)["makespan"], simulate(clustered, cfg)["makespan"])     # 24.0 31.0

# The same number without an event queue (Theorem 3), and the fluid lower bound (Theorem 4)
print(deterministic_runtime(clustered, producers=1, buffer=4, latency=1.0, step_time=1.0))  # 31.0
print(fluid_runtime(clustered, rate=1.0, capacity=5, step_time=1.0))                        # 30.0

# What the histogram alone allows (Theorem 5)
print(histogram_envelope(clustered, rate=1.0, capacity=5, step_time=1.0))
```

A demand trace stored as a CSV file with a `states` column can be run from the command line. The
output contains the Monte Carlo summary, the exact deterministic runtime, the fluid bracket and the
histogram envelope:

```
python -m traceq.cli data/traces/hubbard_6x4_TT.csv -K 11 -B 4 --trials 256
```

## Reproducing the paper

```
python scripts/run_theory.py            # data/results/theory.json
python scripts/run_robustness.py        # data/results/robustness.json
python scripts/build_assets.py          # generated/*.tex and data/results/headline_numbers.json
python scripts/make_figures.py          # figures/*.pdf, figures/*.png
python scripts/verify_artifacts.py      # summary statistics from raw realisations
python scripts/audit_replay.py          # stored realisations through both simulators
```

The main Monte Carlo study is stored in `data/results/study.json` and
`data/results/raw_experiments.csv.gz`. To regenerate and compare it without overwriting anything:

```
python scripts/run_study.py --output-root ../fresh-study
python scripts/verify_full_reproduction.py ../fresh-study
```

All random streams are keyed by experiment family, realisation index and producer, so a rerun
reproduces every realisation exactly.

## Data

* `data/traces/hubbard_6x4_{JW,TT}.csv`: one row per scheduled step (`step`, `states`).
  The JSON files next to them hold the Pauli strings, coefficients, groups and scheduling choices.
* `data/traces/TT_order_*.csv`: four orderings of the ternary-tree demand multiset.
* `data/external/barc/*.u8.xz`: third-party traces, one unsigned byte per step, xz-compressed;
  read them with `traceq.external.load_trace`.
* `data/results/raw_experiments.csv.gz`, `raw_theory_runs.csv.gz`, `raw_robustness_runs.csv.gz`:
  one row per realisation (`experiment`, `run`, `seed`, `makespan`, ...).
* `data/results/study.json`: per-batch configuration, statistics and workload signatures.
* `data/results/theory.json`: closed-form values for every batch, the separation family, the
  histogram envelope, the ordering-by-storage control, stall profiles and the policy table.
* `data/results/robustness.json`: asynchronous progress, synthesis-chain length, staged service
  laws and the 64 external traces.

## Scope

The workload is a Pauli-rotation demand proxy, not a synthesised and routed circuit. The supply law
is a declared scenario (acceptance 0.08, nine-cycle attempts), not a calibrated producer. The
external traces are layer-level T counts of transpiled circuits without placement or routing. The
closed form is exact for constant equal latencies and is an estimator of the mean otherwise. It
carries no tail, residency or deadline information. See the limitations section of the paper.

## Licence and citation

Code is released under the MIT licence (`LICENSE`). Data, traces and figures produced for this paper
are released under CC BY 4.0 (`LICENSE-DATA.md`). The traces in `data/external/barc/` come from the
BARC artifact of Ye, Khan and Liang and keep their Apache 2.0 licence (`NOTICE`). Please cite the
paper; `CITATION.cff` has the details.

Contact: Aishwarya Vishwakarma, Department of Quantum Matter Physics, University of Geneva
(Aishwarya.Vishwakarma@unige.ch).
