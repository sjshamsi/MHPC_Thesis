"""
Shared helpers for the Well dataset registry (`well_dataset_registry.json`).

The registry is written by `build_well_registry.py` and read by, among other things,
`count_WALRUS_flops.py`. For each The Well dataset, It stores facts that affect a
WALRUS run's token counts and cost: Resolution, per-axis boundary conditions, the
model-format channel count, and per-split `(n_steps, n_traj)` groups. Window counts
are not stored, because they depend on
`n_steps_input` / `n_steps_output` / `min_dt_stride`, and `n_windows` derives them later.

`walrus.utils.flops.forward_flops` uses an object called `BatchGeometry` to price out
the flop usage. In this file, `dataset_geometry` turns a registry entry into the  the
`BatchGeometry` object also. This means that `build_well_registry.py --verify` can
checks it against the geometry `FlopAccountant` observes on a real batch, so the
estimator and the check share one code path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

from the_well.data.utils import raw_steps_to_possible_sample_t0s

from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic
from walrus.utils.flops import N_BC_CHANNELS, BatchGeometry, analytic_token_shape

SCHEMA_VERSION = 1
REPO_DIR = Path(__file__).resolve().parents[2]  # Tools/flop_planner/ -> repo root
CONFIG_DIR = REPO_DIR / "walrus" / "configs"
DEFAULT_REGISTRY = Path(__file__).resolve().with_name("well_dataset_registry.json")
SPLITS = ("train", "valid", "test")
MAX_D = 3  # IsotropicModel pads every sample to 3 spatial axes


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> Dict:
    with open(path) as f:
        registry = json.load(f)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version {registry.get('schema_version')}, "
            f"expected {SCHEMA_VERSION}; rebuild it with build_well_registry.py"
        )
    return registry


def save_registry(registry: Dict, path: Path | str = DEFAULT_REGISTRY) -> None:
    with open(path, "w") as f:
        json.dump(registry, f, indent=1, sort_keys=False)
        f.write("\n")


def n_frames(split: Dict) -> int:
    """Σ n_traj × n_steps over a split entry."""
    return sum(g["n_traj"] * g["n_steps"] for g in split["steps"])


def n_windows(split: Dict, n_steps_input: int, n_steps_output: int, min_dt_stride: int) -> int:
    """Training windows (= len(dataset)), computed exactly as WellDataset._build_metadata."""
    return sum(
        g["n_traj"]
        * raw_steps_to_possible_sample_t0s(
            g["n_steps"], n_steps_input, n_steps_output, min_dt_stride
        )
        for g in split["steps"]
    )


def padded_resolution(entry: Dict) -> Tuple[int, ...]:
    res = tuple(entry["resolution"])
    return res + (1,) * (MAX_D - len(res))


def periodic_flags(entry: Dict) -> Tuple[bool, ...]:
    """Per-axis periodicity, padded to MAX_D axes.

    Padded (singleton) axes are periodic, as InflatedWellDataset._pad_to_higher_d sets
    them; it never matters for cost, because singleton axes are neither padded nor strided.
    """
    flags = [axis[0] == "periodic" for axis in entry["bcs"]]
    return tuple(flags + [True] * (MAX_D - len(flags)))


def token_shape(entry: Dict) -> Tuple[int, ...]:
    return analytic_token_shape(padded_resolution(entry), periodic_flags(entry))


def encoder_shapes_from_tokens(
    token_shape: Sequence[int],
    stride1: Sequence[int],
    stride2: Sequence[int],
    kernel1: Sequence[int],
    kernel2: Sequence[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Invert the encoder to `(padded_in_shape, mid_shape)` for a wanted token grid.

    The patch jitterer pads the input to whatever makes the two `padding=0` convs land
    on the target token count, so the padded shape is recovered by inverting
    `conv_out_size` twice rather than by modelling the padding rule. Planning only:
    `FlopAccountant` observes the real shape instead.
    """
    mid, padded = [], []
    for tokens, s1, s2, k1, k2 in zip(token_shape, stride1, stride2, kernel1, kernel2):
        if tokens == 1:
            mid.append(1)
            padded.append(1)
            continue
        mid_size = (tokens - 1) * s2 + k2
        mid.append(mid_size)
        padded.append((mid_size - 1) * s1 + k1)
    return tuple(padded), tuple(mid)


def dataset_geometry(
    entry: Dict,
    kernel1: Sequence[int],
    kernel2: Sequence[int],
    n_time_in: int,
    batch_size: int,
) -> BatchGeometry:
    """The BatchGeometry one micro-batch of this dataset has, without loading any data.

    Strides come from the same `choose_kernel_size_deterministic` the model calls; the
    padded encoder input is recovered by inverting the two convs from the token grid
    (`encoder_shapes_from_tokens`), which is how the patch jitterer's padding works out.
    """
    shape = padded_resolution(entry)
    periodic = periodic_flags(entry)
    kernels = choose_kernel_size_deterministic(shape)
    stride1 = tuple(int(k[0]) for k in kernels)
    stride2 = tuple(int(k[1]) for k in kernels)
    tokens = analytic_token_shape(shape, periodic)
    encoder_in, _ = encoder_shapes_from_tokens(tokens, stride1, stride2, kernel1, kernel2)
    return BatchGeometry(
        encoder_in_shape=encoder_in,
        stride1=stride1,
        stride2=stride2,
        periodic=periodic,
        n_fields_in=entry["n_channels_in"] + N_BC_CHANNELS,
        n_fields_out=entry["n_channels_in"],
        n_time_in=n_time_in,
        batch_size=batch_size,
        token_shape=tokens,
        dataset_name=entry["name"],
    )


def unique_model_tokens(entry: Dict, split: str = "train") -> int:
    """`T_unique_model` for one dataset: Σ n_traj × n_steps × N_spatial over a split.

    Same per-axis-BC rule as `walrus.utils.flops.unique_model_tokens` (which feeds the
    W&B `repetition_factor`); the two agree exactly on all 18 datasets on disk.
    """
    return n_frames(entry["splits"][split]) * math.prod(token_shape(entry))
