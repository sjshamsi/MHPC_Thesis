"""
Build (or check) the Well dataset registry (containing information like the
number of steps, number of trajectories, axis bounfry conditions, etc.).
This is used among other things, as a cheap database by `count_WALRUS_flops.py`.

It scans each dataset's HDF5 metadata through `InflatedWellDataset` (reader object
used while training). The object also pads 2D datasets to 3D as in training)
and writes one JSON entry per dataset to `/well_dataset_registry.json`.
It has the following information:

    resolution, per-axis boundary conditions, native and 3D-padded field names,
    n_channels_in (model-format channels the encoder sees, excluding the 3 BC flags),
    token grid, and per split: files, bytes on disk, (n_steps, n_traj) groups, and a
    per-file list of (name, n_traj, n_steps, bytes).

While no frames are read, each file's metadata is read so a full scan of $WORK takes
several minutes. Rebuild after downloading or deleting datasets.

`--verify` pushes one real window per dataset through the real pipeline
(datamodule -> formatter -> tiny IsotropicModel) and checks that the geometry observed
by the `FlopAccountant` object equals the one which `well_registry.dataset_geometry`
derives from the registry, and that both price to the same FLOPs. Run after every rebuild.

Usage (from the repo root, walrus venv; --verify-only also via Runs/verify_registry/):
    module purge; module load python/3.11.7; module load cuda/12.6
    source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env2/bin/activate
    python Tools/flop_planner/build_well_registry.py                    # scan everything under well_base_path
    python Tools/flop_planner/build_well_registry.py --datasets MHD_64  # rescan one, keep the rest
    python Tools/flop_planner/build_well_registry.py --path /some/external/dataset_dir
    python Tools/flop_planner/build_well_registry.py --verify-only      # check the existing registry
    python Tools/flop_planner/build_well_registry.py --csv registry.csv # also write a flat summary
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import h5py
from omegaconf import OmegaConf
from the_well.data.datasets import BoundaryCondition
from the_well.data.utils import flatten_field_names
from well_registry import (
    CONFIG_DIR,
    DEFAULT_REGISTRY,
    REPO_DIR,
    SCHEMA_VERSION,
    SPLITS,
    dataset_geometry,
    load_registry,
    n_frames,
    save_registry,
    token_shape,
)

from walrus.data.inflated_dataset import InflatedWellDataset

warnings.filterwarnings("ignore")


def default_root() -> str:
    server = OmegaConf.load(CONFIG_DIR / "server" / "leonardo.yaml")
    return str(server.data.well_base_path)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ------------------------------------------------------------------------ scanning
def open_split(path: Path, split: str):
    if not (path / "data" / split).is_dir():
        return None
    try:
        return InflatedWellDataset(path=str(path), well_split_name=split, pad_cartesian_data_to_d=3)
    except AssertionError:  # "No HDF5 files found"
        return None


def axis_bcs(ds) -> list:
    """Per-axis (lo, hi) BC names for the dataset's real axes, via the_well's own reader."""
    with h5py.File(ds.files_paths[0], "r") as f:
        bcs = ds._padding_bcs(f, {}, 0, 0, 0, 1)  # cache, sample, t, n_steps, dt are unused
    n = ds.original_metadata.n_spatial_dims
    return [[BoundaryCondition(int(v)).name.lower() for v in axis] for axis in bcs[:n]]


def split_entry(ds) -> dict:
    md = ds.original_metadata
    groups = Counter()
    files = []
    # files_paths, n_trajectories_per_file and n_steps_per_trajectory are filled in the same
    # per-file loop of WellDataset._build_metadata, so they line up index by index.
    for path, n_traj, n_steps in zip(ds.files_paths, md.n_trajectories_per_file, md.n_steps_per_trajectory):
        groups[n_steps] += n_traj
        files.append({"name": os.path.basename(path), "n_traj": n_traj, "n_steps": n_steps,
                      "bytes": os.path.getsize(path)})
    return {
        "n_files": md.n_files,
        "bytes": sum(f["bytes"] for f in files),
        "steps": [{"n_steps": s, "n_traj": t} for s, t in sorted(groups.items())],
        "files": files,
    }


def scan_dataset(path: Path) -> dict:
    splits = {s: open_split(path, s) for s in SPLITS}
    present = {s: ds for s, ds in splits.items() if ds is not None}
    if not present:
        raise FileNotFoundError(f"no Well HDF5 files under {path}/data/{{{','.join(SPLITS)}}}")
    ref = present.get("train") or next(iter(present.values()))
    native, padded = ref.original_metadata, ref.metadata
    entry = {
        "name": native.dataset_name,
        "path": str(path),
        "n_spatial_dims": native.n_spatial_dims,
        "grid_type": native.grid_type,
        "resolution": list(native.spatial_resolution),
        "bcs": axis_bcs(ref),
        "bc_types": sorted(native.boundary_condition_types),
        "field_names": {str(k): v for k, v in native.field_names.items() if v},
        "constant_field_names": {str(k): v for k, v in native.constant_field_names.items() if v},
        # What MixedWellDataset hands the encoder: flatten_field_names on the padded metadata
        # (fields then constants), i.e. len(field_indices) before the 3 BC flags are appended.
        "model_field_names": flatten_field_names(padded),
        "n_channels_in": len(flatten_field_names(padded)),
        "splits": {s: split_entry(ds) for s, ds in present.items()},
    }
    entry["token_grid"] = list(token_shape(entry))
    return entry


def discover(root: Path) -> list:
    return sorted(p for p in root.iterdir() if (p / "data").is_dir())


# ------------------------------------------------------------------------ verify
VERIFY_OVERRIDES = [
    "server=leonardo",
    "data=Leonardo_pretrain9_2d",  # replaced below by one dataset at a time
    "data/field_index_map_override=full_well_field_index",
    "distribution=local",
    "logger=none",
    "data_workers=0",
    "automatic_setup=False",
    "data.module_parameters.batch_size=1",
    "data.module_parameters.n_steps_input=2",
    "data.module_parameters.n_steps_output=1",
    # Smallest width that builds: space RoPE has dim hidden_dim / heads / 4, and
    # lr_rope_temporary.py divides by (dim - 2), so hidden_dim=128 with 16 heads crashes.
    "model.hidden_dim=256",
    "model.intermediate_dim=64",
    "model.groups=16",
    "model.processor_blocks=1",
    "model.processor.space_mixing.mlp_dim=512",
    "model/processor/space_mixing=full_spatial_attention",
    "model.processor.space_mixing.num_heads=16",
    "model.processor.time_mixing.num_heads=16",
    "model.override_dimensionality=0",
    "model.causal_in_time=True",
    "++model.use_periodic_fixed_jitter=True",
]


def per_axis(geom) -> list:
    """Axis-order-free view of a geometry: RandomRotation90 may permute the axes.

    The BC flag of a singleton axis is dropped: such an axis is never padded or strided,
    so its flag cannot affect cost, and non-cartesian datasets (planetswe) leave it
    non-periodic where cartesian ones mark it periodic.
    """
    return sorted(
        (size, s1, s2, periodic if size > 1 else None, tokens)
        for size, s1, s2, periodic, tokens in zip(
            geom.encoder_in_shape, geom.stride1, geom.stride2, geom.periodic, geom.token_shape
        )
    )


def verify(registry: dict, names: list) -> bool:
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
    from walrus.utils.flops import FlopAccountant, forward_flops

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=VERIFY_OVERRIDES)
    OmegaConf.set_struct(cfg, False)
    fmt = ChannelsFirstWithTimeFormatter()

    print(f"{'dataset':<40}{'ch':>4}{'tokens':>16}{'encoder in':>18}{'geometry':>10}{'FLOPs':>8}")
    ok = True
    for name in names:
        entry = registry["datasets"][name]
        info = {"include_filters": [], "exclude_filters": []}
        if Path(entry["path"]).parent != Path(registry["root"]):
            info["path"] = entry["path"]
        cfg.data.module_parameters.well_dataset_info = {name: info}
        try:
            dm = instantiate(
                cfg.data.module_parameters, world_size=1, rank=0, data_workers=0,
                well_base_path=cfg.data.well_base_path,
                field_index_map_override=cfg.data.get("field_index_map_override", {}),
                transform=cfg.data.get("transform", None),
            )
            mixed = dm.train_dataset
            n_states = max(mixed.field_to_index_map.values()) + 1
            model = instantiate(cfg.model, n_states=n_states).eval()
            batch = mixed[0]
            x, labels, bcs = fmt.process_input(batch, causal_in_time=True, predict_delta=True, train=True)[0][:3]
            acc = FlopAccountant(model, world_size=1)
            with torch.no_grad():
                model(x, labels, bcs.tolist(), metadata=batch["metadata"])
            observed = acc._observed
            acc.detach()
        except Exception as e:  # report and keep going: one broken dataset shouldn't hide the rest
            print(f"{name:<40}  ERROR {type(e).__name__}: {e}")
            ok = False
            continue
        derived = dataset_geometry(
            entry, acc.shape.encoder_kernel1, acc.shape.encoder_kernel2,
            n_time_in=observed.n_time_in, batch_size=observed.batch_size,
        )
        geom_ok = (observed.n_fields_in == derived.n_fields_in
                   and per_axis(observed) == per_axis(derived))
        # Every granular term must match, not just the total.
        flops_ok = forward_flops(acc.shape, observed) == forward_flops(acc.shape, derived)
        ok &= geom_ok and flops_ok
        print(f"{name:<40}{observed.n_fields_in - 3:>4}{str(observed.token_shape):>16}"
              f"{str(observed.encoder_in_shape):>18}{'ok' if geom_ok else 'FAIL':>10}"
              f"{'exact' if flops_ok else 'FAIL':>8}")
        if not geom_ok:
            print(f"    observed: ch={observed.n_fields_in - 3} {per_axis(observed)}")
            print(f"    registry: ch={derived.n_fields_in - 3} {per_axis(derived)}")
    return ok


# ------------------------------------------------------------------------ output
def write_csv(registry: dict, path: str) -> None:
    cols = ["name", "n_spatial_dims", "resolution", "bcs", "n_channels_in", "token_grid",
            "train_traj", "train_frames", "train_GB", "valid_traj", "test_traj", "total_GB"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in registry["datasets"].values():
            sp = e["splits"]
            traj = {s: sum(g["n_traj"] for g in sp[s]["steps"]) if s in sp else "" for s in SPLITS}
            w.writerow([
                e["name"], e["n_spatial_dims"], "x".join(map(str, e["resolution"])),
                ", ".join("/".join(a) for a in e["bcs"]), e["n_channels_in"],
                "x".join(map(str, e["token_grid"])), traj["train"],
                n_frames(sp["train"]) if "train" in sp else "",
                round(sp["train"]["bytes"] / 1e9, 1) if "train" in sp else "",
                traj["valid"], traj["test"],
                round(sum(s["bytes"] for s in sp.values()) / 1e9, 1),
            ])
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="datasets root (default: well_base_path in configs/server/leonardo.yaml)")
    p.add_argument("--datasets", nargs="+", default=None, help="scan only these dataset dirs under --root; other entries are kept")
    p.add_argument("--path", nargs="+", default=[], help="extra dataset dirs outside --root (external Well-format data)")
    p.add_argument("--out", default=str(DEFAULT_REGISTRY), help="registry file (default: %(default)s)")
    p.add_argument("--csv", default=None, help="also write a flat per-dataset summary CSV here")
    p.add_argument("--verify", action="store_true", help="after building, check the geometry on one real batch per dataset")
    p.add_argument("--verify-only", action="store_true", help="skip scanning; verify the existing registry (or --datasets)")
    args = p.parse_args()

    root = Path(args.root or default_root())
    if args.verify_only:
        registry = load_registry(args.out)
        names = args.datasets or list(registry["datasets"])
        sys.exit(0 if verify(registry, names) else 1)

    registry = load_registry(args.out) if Path(args.out).exists() and (args.datasets or args.path) else {"datasets": {}}
    paths = [root / d for d in args.datasets] if args.datasets else ([] if args.path else discover(root))
    paths += [Path(x) for x in args.path]

    for path in paths:
        try:
            entry = scan_dataset(path)
        except (FileNotFoundError, OSError, KeyError) as e:
            print(f"skip {path}: {e}")
            continue
        registry["datasets"][path.name] = entry
        tr = entry["splits"].get("train")
        print(f"{path.name:<40}{'x'.join(map(str, entry['resolution'])):>14}  ch={entry['n_channels_in']:<3}"
              f" tokens={'x'.join(map(str, entry['token_grid'])):<10}"
              f" train traj={sum(g['n_traj'] for g in tr['steps']) if tr else '-'}")

    registry.update({
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "git_sha": git_sha(),
    })
    registry["datasets"] = dict(sorted(registry["datasets"].items()))
    # Header first, datasets last, so the file reads top-down.
    registry = {k: registry[k] for k in ("schema_version", "built_at", "root", "git_sha", "datasets")}
    save_registry(registry, args.out)
    print(f"wrote {args.out} ({len(registry['datasets'])} datasets)")
    if args.csv:
        write_csv(registry, args.csv)
    if args.verify:
        names = [p.name for p in paths if p.name in registry["datasets"]]
        sys.exit(0 if verify(registry, names) else 1)


if __name__ == "__main__":
    main()
