"""
Training FLOPs of a WALRUS job from its run script, without running it.

Reads a SLURM run script (Hydra overrides after `train.py`, plus SLURM settinsg like `#SBATCH --nodes` and `--gpus-per-node`), and composes the same config train.py would, and prices every epoch:

- per micro-batch, per dataset: `walrus.utils.flops.forward_flops` / `training_step_flops`
  on the geometry `well_registry.dataset_geometry` derives from the dataset registry
  (`build_well_registry.py`; its --verify checks that geometry against real batches);
- how many micro-batches each dataset gets: the expectation (uniform over datasets), or
  with --exact, a replay of BatchedMultisetSampler's dataset-choice generator.

The FLOP basis is the one FlopAccountant logs to W&B: fwd + bwd (bwd = 2x fwd), with
gradient-checkpointing recompute reported separately. "total" here is measured
(useful + recompute); W&B's `cumulative_flops` is the useful part. Validation is not included.

Examples (from the repo root, walrus venv):
    python Tools/flop_planner/count_WALRUS_flops.py <run_script.sh>
    python Tools/flop_planner/count_WALRUS_flops.py <run_script.sh> --exact
    python Tools/flop_planner/count_WALRUS_flops.py <run_script.sh> -H 1408 -L 40 -M 5632 --nodes 4
    python Tools/flop_planner/count_WALRUS_flops.py walrus/run_scripts/pretrain_example_distributed_walrus.sh
    python Tools/flop_planner/count_WALRUS_flops.py --override data=Leonardo_pretrain9_2d trainer.max_epoch=10 --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from well_registry import (  # noqa: E402
    CONFIG_DIR,
    DEFAULT_REGISTRY,
    dataset_geometry,
    load_registry,
    n_windows,
    unique_model_tokens,
)

# Shell placeholders in generated run scripts, replaced as pt9_substrate.check_composes does.
SHELL_SUBSTITUTIONS = {"$AUTO_RESUME": "True", "$STAGE_DIR": "/scratch_local/flop_estimate"}


# ------------------------------------------------------------------------ job resolution
def parse_run_script(path: Path):
    """Return (hydra overrides, nodes, gpus_per_node) from a SLURM run script."""
    
    text = path.read_text()
    nodes = gpus = None
    for line in text.splitlines():
        m = re.match(r"#SBATCH\s+(?:--nodes[= ]|-N\s*)(\d+)", line)
        if m:
            nodes = int(m.group(1))
        m = re.match(r"#SBATCH\s+--gpus-per-node[= ](?:\w+:)?(\d+)", line)
        if m:
            gpus = int(m.group(1))

    # Join backslash continuations, then find the logical line that launches train.py.
    logical = re.sub(r"\\\n", " ", text).splitlines()
    launch = [ln for ln in logical if re.search(r"\btrain\.py\b", ln) and not ln.lstrip().startswith("#")]
    if not launch:
        raise ValueError(f"{path}: no line launching train.py found")
    
    tokens = shlex.split(launch[0], comments=True, posix=True)
    overrides = tokens[tokens.index(next(t for t in tokens if t.endswith("train.py"))) + 1:]
    cleaned = []
    for o in overrides:
        if override_key(o).startswith("hydra."):  # run dirs ($SLURM_JOB_ID, ${now:...}); no effect on FLOPs
            continue
        for k, v in SHELL_SUBSTITUTIONS.items():
            o = o.replace(k, v)
        if "$" in o:
            raise ValueError(f"{path}: unresolved shell variable in override {o!r}")
        cleaned.append(o)

    return cleaned, nodes, gpus


def override_key(o: str) -> str:
    return o.split("=", 1)[0].lstrip("+~")


def dedupe(overrides):
    """Keep the last value per key, so later (CLI) overrides win over the script's."""
    last = {override_key(o): o for o in overrides}
    return list(last.values())


def compose_config(overrides):
    from hydra import compose, initialize_config_dir

    # Paths and logging don't affect FLOPs; pin the server so upstream scripts (server=gpuxl) compose.
    keep = [o for o in overrides if not override_key(o).startswith("hydra.") and override_key(o) != "server"]
    keep = dedupe(keep + ["server=leonardo"])
    
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=keep), keep


def model_shape(cfg):
    """ModelShape (and param count) from the composed model config, built on the meta device."""
    import torch
    from hydra.utils import instantiate

    from walrus.utils.flops import ModelShape

    fmap = cfg.data.get("field_index_map_override") or {}
    # n_states only sizes the stem's weights (params), never the FLOPs: the encoder convolves
    # over the dataset's own fields. With the 3-entry bc_only map, the real value depends on
    # which Well datasets are on disk; 67 is the full Well map.
    n_states = max(fmap.values()) + 1 if len(fmap) > 3 else 67
    with torch.device("meta"):
        model = instantiate(cfg.model, n_states=n_states)
    return ModelShape.from_model(model), sum(p.numel() for p in model.parameters()), n_states


def sampling_layout(cfg, world_size: int, gpus_per_node: int):
    """Distinct dataset-choice streams, mirroring training.py:358-376 and configure_distribution."""
    dist = cfg.distribution.distribution_type.lower()
    if dist == "fsdp":
        sync = world_size
    elif dist == "hsdp":
        sync = int(cfg.distribution.get("local_size") or gpus_per_node)
    else:  # ddp, local
        sync = 1
    strategy = cfg.trainer.get("sampling_rank_strategy", "node")
    if strategy == "gpu":
        streams, per_stream = world_size, 1
    elif strategy == "node":
        streams, per_stream = world_size // sync, sync
    else:  # "world"
        streams, per_stream = 1, world_size
    return dist, streams, per_stream, strategy


# ------------------------------------------------------------------------ micro-batch counts
def replay_epoch(seed, dset_batch, windows, max_samples, recycle):
    """Micro-batches per dataset for one sampler pass: BatchedMultisetSampler.__iter__
    (mixed_dset_sampler.py:167-210), with sub-sampler exhaustion modelled from window counts."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    k = len(windows)
    remaining = list(windows)
    counts = [0] * k
    choices = list(range(k))
    count = 0
    while choices:
        count += 1
        i = int(torch.randint(0, len(choices), size=(1,), generator=g).item())
        d = choices[i]
        if remaining[d] < dset_batch[d]:  # StopIteration part-way through the batch
            if recycle:
                count -= 1
                remaining[d] = windows[d]  # fresh RandomSampler pass
            else:
                choices.pop(i)
            continue
        remaining[d] -= dset_batch[d]
        counts[d] += 1
        if count >= max_samples:
            break
    return counts


def microbatch_counts(datasets, epochs, max_samples, world_size, streams, per_stream,
                      distributed, recycle, exact):
    """Micro-batches per dataset over the job, summed over all ranks."""
    k = len(datasets)
    if not exact:
        return [max_samples * world_size * len(epochs) / k] * k
    dset_batch = [d["batch"] for d in datasets]
    windows = [d["windows"] for d in datasets]
    totals = [0] * k
    for epoch in epochs:
        sampler_epoch = epoch if distributed else 0  # set_epoch is only called when distributed
        for rank in range(streams):
            seed = 100 * sampler_epoch + 10 * 0 + rank  # sampler seed is never passed: 0
            for d, c in enumerate(replay_epoch(seed, dset_batch, windows, max_samples, recycle)):
                totals[d] += c * per_stream
    return totals


# ------------------------------------------------------------------------ main
def build_parser():
    p = argparse.ArgumentParser(
        prog="count_WALRUS_flops.py",
        description="Training FLOPs of a WALRUS job from its run script (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples (from the repo root, walrus venv):")[1],
    )
    p.add_argument("script", nargs="?", default=None, help="SLURM run script that launches train.py")
    w = p.add_argument_group("what-if overrides (applied after the script's)")
    w.add_argument("-H", "--hidden-dim", type=int, help="model.hidden_dim")
    w.add_argument("-L", "--processor-blocks", type=int, help="model.processor_blocks")
    w.add_argument("-M", "--mlp-dim", type=int, help="model.processor.space_mixing.mlp_dim")
    w.add_argument("--data", help="data config, e.g. Leonardo_pretrain9_2d")
    w.add_argument("--batch-size", type=int, help="data.module_parameters.batch_size")
    w.add_argument("--n-steps-input", type=int, help="data.module_parameters.n_steps_input")
    w.add_argument("--max-samples", type=int, help="data.module_parameters.max_samples (micro-batches per rank per epoch)")
    w.add_argument("--grad-acc-steps", type=int, help="trainer.grad_acc_steps")
    w.add_argument("--epochs", type=int, help="trainer.max_epoch")
    w.add_argument("--distribution", choices=["local", "ddp", "fsdp", "hsdp"], help="distribution=")
    w.add_argument("--nodes", type=int, help="SLURM nodes (default: the script's #SBATCH, else 1)")
    w.add_argument("--gpus-per-node", type=int, help="GPUs per node (default: the script's #SBATCH, else 4)")
    w.add_argument("--override", nargs="+", default=[], metavar="KEY=VALUE", help="any other Hydra override")
    s = p.add_argument_group("span and method")
    s.add_argument("--start-epoch", type=int, default=1, help="first epoch to price (default: 1)")
    s.add_argument("--end-epoch", type=int, default=None, help="last epoch to price (default: max_epoch)")
    s.add_argument("--exact", action="store_true",
                   help="replay the sampler's dataset draws instead of assuming 1/K per dataset")
    s.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="dataset registry (default: %(default)s)")
    s.add_argument("--json", action="store_true", help="print JSON instead of tables")
    return p


def cli_overrides(a):
    pairs = [
        ("model.hidden_dim", a.hidden_dim), ("model.processor_blocks", a.processor_blocks),
        ("model.processor.space_mixing.mlp_dim", a.mlp_dim), ("data", a.data),
        ("data.module_parameters.batch_size", a.batch_size),
        ("data.module_parameters.n_steps_input", a.n_steps_input),
        ("data.module_parameters.max_samples", a.max_samples),
        ("trainer.grad_acc_steps", a.grad_acc_steps), ("trainer.max_epoch", a.epochs),
        ("distribution", a.distribution),
    ]
    out = [f"{k}={v}" for k, v in pairs if v is not None]
    # A new hidden_dim without an mlp_dim means FullAttention's default, 4 * hidden_dim.
    if a.hidden_dim and not a.mlp_dim:
        out.append(f"model.processor.space_mixing.mlp_dim={4 * a.hidden_dim}")
    return out + a.override


def fmt_flops(x):
    for unit, scale in (("EF", 1e18), ("PF", 1e15), ("TF", 1e12), ("GF", 1e9)):
        if x >= scale:
            return f"{x / scale:.3f} {unit}"
    return f"{x:.3g} F"


def fmt_count(x):
    for unit, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if x >= scale:
            return f"{x / scale:.2f} {unit}"
    return f"{x:.0f}"


def main():
    parser = build_parser()
    a = parser.parse_args()
    if a.script is None and not cli_overrides(a):
        parser.error("give a run script, or overrides (at least --data or --override data=...)")

    script_overrides, nodes, gpus = ([], None, None)
    if a.script:
        script_overrides, nodes, gpus = parse_run_script(Path(a.script))
    nodes = a.nodes or nodes or 1
    gpus = a.gpus_per_node or gpus or 4
    cfg, used_overrides = compose_config(script_overrides + cli_overrides(a))

    shape, n_params, n_states = model_shape(cfg)
    mp = cfg.data.module_parameters
    max_epoch = int(cfg.trainer.max_epoch)
    epochs = range(a.start_epoch, (a.end_epoch or max_epoch) + 1)  # training.py: 1-indexed, inclusive
    if len(epochs) == 0:
        parser.error(f"empty epoch range {a.start_epoch}..{a.end_epoch or max_epoch}")
    dist, streams, per_stream, strategy = sampling_layout(cfg, nodes * gpus, gpus)
    world_size = 1 if dist == "local" else nodes * gpus
    if dist == "local":
        streams, per_stream = 1, 1
    grad_acc = int(cfg.trainer.grad_acc_steps)
    max_samples = int(mp.max_samples)
    for knob in ("restrict_train_num_trajectories", "restrict_train_num_samples"):
        if mp.get(knob) is not None:
            print(f"warning: {knob}={mp.get(knob)} is ignored; windows/unique tokens use the full train split",
                  file=sys.stderr)

    # --- per-dataset geometry and per-micro-batch cost
    from walrus.utils.flops import forward_flops, stage_totals, training_step_flops

    registry = load_registry(a.registry)
    if Path(registry["root"]) != Path(cfg.data.well_base_path):
        print(f"warning: registry root {registry['root']} != well_base_path {cfg.data.well_base_path}",
              file=sys.stderr)
    info = mp.well_dataset_info
    missing = [n for n in info if n not in registry["datasets"]]
    if missing:
        sys.exit("not in the registry: " + ", ".join(missing) + "\nadd them with:\n"
                 f"  python Tools/flop_planner/build_well_registry.py --datasets {' '.join(missing)}\n"
                 "(or --path <dir> for datasets outside well_base_path)")

    datasets = []
    for name, dinfo in info.items():
        entry = registry["datasets"][name]
        if not Path(entry["path"]).exists():
            print(f"warning: {name}: registry path {entry['path']} no longer exists", file=sys.stderr)
        # Same flooring as multidataset.py:259 and mixed_dset_sampler.py:184-185.
        n_in = max(1, int(int(mp.n_steps_input) * dinfo.get("step_downsample_factor", 1)))
        batch = max(1, int(int(mp.batch_size) * dinfo.get("batch_downsample_factor", 1)))
        geom = dataset_geometry(entry, shape.encoder_kernel1, shape.encoder_kernel2, n_in, batch)
        step = training_step_flops(forward_flops(shape, geom), shape)
        datasets.append({
            "name": name,
            "resolution": entry["resolution"],
            "channels": entry["n_channels_in"],
            "token_grid": list(geom.token_shape),
            "n_steps_input": n_in,
            "batch": batch,
            "windows": n_windows(entry["splits"]["train"], n_in, int(mp.n_steps_output), int(mp.min_dt_stride)),
            "unique_tokens": unique_model_tokens(entry),
            "tokens_per_microbatch": batch * n_in * math.prod(geom.token_shape),
            "step": step,  # granular fwd+bwd, "useful", "recompute"
            "stages": stage_totals(step),  # encoder/processor/decoder/total, fwd+bwd
            "flops_measured": step["useful"] + step["recompute"],
        })

    counts = microbatch_counts(
        datasets, epochs, max_samples, world_size, streams, per_stream,
        distributed=dist != "local", recycle=bool(mp.get("recycle_datasets", True)), exact=a.exact,
    )

    # --- totals over the job (all ranks)
    def total(fn):
        return sum(c * fn(d) for c, d in zip(counts, datasets))

    job = {
        "microbatches": sum(counts),
        "tokens_seen": total(lambda d: d["tokens_per_microbatch"]),
        "flops_total": total(lambda d: d["flops_measured"]),
        "flops_useful": total(lambda d: d["step"]["useful"]),
        "flops_recompute": total(lambda d: d["step"]["recompute"]),
        "flops_encoder": total(lambda d: d["stages"]["encoder"]),
        "flops_processor": total(lambda d: d["stages"]["processor"]),
        "flops_decoder": total(lambda d: d["stages"]["decoder"]),
    }
    # Every granular term is forward x 3, so this is also the forward-only fraction.
    job["forward_enc_dec_fraction"] = (job["flops_encoder"] + job["flops_decoder"]) / job["flops_useful"]
    n_ep = len(epochs)
    opt_steps = n_ep * max_samples // grad_acc
    unique = sum(d["unique_tokens"] for d in datasets)
    job |= {
        "epochs": n_ep,
        "optimizer_steps": opt_steps,
        "flops_per_epoch": job["flops_total"] / n_ep,
        "flops_per_optimizer_step": job["flops_total"] / opt_steps if opt_steps else float("nan"),
        "tokens_per_epoch": job["tokens_seen"] / n_ep,
        "unique_model_tokens": unique,
        "repetition_factor": job["tokens_seen"] / unique,
    }

    run = {
        "script": a.script,
        "overrides": used_overrides,
        "shape": {"hidden_dim": shape.hidden_dim, "processor_blocks": shape.processor_blocks,
                  "mlp_dim": shape.mlp_dim, "intermediate_dim": shape.intermediate_dim,
                  "causal_in_time": shape.causal_in_time,
                  "gradient_checkpointing_freq": shape.gradient_checkpointing_freq},
        "params": n_params, "n_states": n_states,
        "data": cfg.data.get("wandb_data_name", ""),
        "distribution": dist, "sampling_rank_strategy": strategy,
        "nodes": nodes, "gpus_per_node": gpus, "world_size": world_size,
        "dataset_streams": streams, "batch_size": int(mp.batch_size),
        "n_steps_input": int(mp.n_steps_input), "max_samples": max_samples,
        "grad_acc_steps": grad_acc, "epoch_range": [epochs[0], epochs[-1]],
        "method": "exact sampler replay" if a.exact else "expected (1/K per dataset)",
    }
    per_dataset = [{
        "name": d["name"], "microbatches": c, "share": c / job["microbatches"],
        "flops_per_microbatch": d["flops_measured"],
        "forward_enc_dec_fraction": (d["stages"]["encoder"] + d["stages"]["decoder"]) / d["stages"]["total"],
        "flops_total": c * d["flops_measured"], "tokens_seen": c * d["tokens_per_microbatch"],
        "unique_tokens": d["unique_tokens"],
        "repetition_factor": c * d["tokens_per_microbatch"] / d["unique_tokens"],
        "resolution": d["resolution"], "channels": d["channels"], "token_grid": d["token_grid"],
        "batch": d["batch"], "n_steps_input": d["n_steps_input"], "train_windows": d["windows"],
    } for c, d in zip(counts, datasets)]

    if a.json:
        print(json.dumps({"run": run, "job": job, "datasets": per_dataset}, indent=2, default=float))
        return

    sh = run["shape"]
    print(f"{a.script or '(overrides only)'}")
    print(f"  model     {sh['hidden_dim']}/{sh['processor_blocks']}/{sh['mlp_dim']} (H/L/M), "
          f"intermediate {sh['intermediate_dim']}, {n_params / 1e6:.1f} M params (n_states={n_states}), "
          f"causal={sh['causal_in_time']}, grad-ckpt freq {sh['gradient_checkpointing_freq']}")
    print(f"  data      {run['data']}: {len(datasets)} datasets, batch {run['batch_size']}, "
          f"T_in {run['n_steps_input']}, max_samples {max_samples}")
    print(f"  compute   {dist} on {nodes} x {gpus} GPUs (world {world_size}), {streams} dataset-choice "
          f"stream(s) [{strategy}], grad_acc {grad_acc}")
    print(f"  epochs    {epochs[0]}..{epochs[-1]} ({n_ep}), {opt_steps:,} optimizer steps; "
          f"counts: {run['method']}")
    print()
    print(f"  total training FLOPs   {fmt_flops(job['flops_total'])}"
          f"   (useful {fmt_flops(job['flops_useful'])} + recompute {fmt_flops(job['flops_recompute'])})")
    print(f"    encoder / processor / decoder   {fmt_flops(job['flops_encoder'])} / "
          f"{fmt_flops(job['flops_processor'])} / {fmt_flops(job['flops_decoder'])}"
          f"   (enc+dec {job['forward_enc_dec_fraction']:.1%} of forward)")
    print(f"  per epoch              {fmt_flops(job['flops_per_epoch'])}")
    print(f"  per optimizer step     {fmt_flops(job['flops_per_optimizer_step'])}")
    print(f"  micro-batches          {job['microbatches']:,.0f} ({job['microbatches'] / n_ep:,.0f}/epoch)")
    print(f"  tokens seen            {fmt_count(job['tokens_seen'])} ({fmt_count(job['tokens_per_epoch'])}/epoch)")
    print(f"  unique model tokens    {fmt_count(unique)} (train split)   repetition {job['repetition_factor']:.2f}x")
    print()
    hdr = (f"  {'dataset':<36}{'res':>13}{'ch':>4}{'tokens':>10}{'BxT':>6}{'µbatches':>11}"
           f"{'TF/µb':>9}{'enc+dec':>9}{'job FLOPs':>13}{'repeat':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for d in per_dataset:
        res = "x".join(map(str, d["resolution"]))
        grid = "x".join(map(str, d["token_grid"]))
        bxt = f"{d['batch']}x{d['n_steps_input']}"
        print(f"  {d['name']:<36}{res:>13}{d['channels']:>4}{grid:>10}{bxt:>6}"
              f"{d['microbatches']:>11,.0f}{d['flops_per_microbatch'] / 1e12:>9.2f}"
              f"{d['forward_enc_dec_fraction']:>9.1%}{fmt_flops(d['flops_total']):>13}"
              f"{d['repetition_factor']:>8.2f}x")


if __name__ == "__main__":
    main()
