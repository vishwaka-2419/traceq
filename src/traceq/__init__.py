"""Trace-aware finite-buffer magic-state provisioning, research reference model."""
__version__ = "0.3.0"
from .model import Config, simulate, simulate_reference, sample_latencies
from .analysis import fluid_buffer_requirement, summarize, certified_violation_bound
from .theory import (deterministic_runtime, fluid_runtime, fluid_overload_intervals,
                     histogram_envelope, cluster_family, cluster_family_runtimes, minimax_floor)
from .asynchronous import rotation_graph, simulate_lanes
from .external import load_trace
