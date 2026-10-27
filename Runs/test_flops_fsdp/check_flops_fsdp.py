"""Check the FSDP FLOP-accounting run of test_flops_fsdp.sh (run at the end of that job).

Compares three independent records of the same 4-epoch run:
  * W&B: the offline history in logs/wandb/offline-run-*-<run id>/run-*.wandb (one per phase);
  * checkpoints: checkpoints/step_N/metadata.pt["flop_accountant"];
  * the planner: Tools/flop_planner/count_WALRUS_flops.py --exact, per epoch, same overrides.

Checks (relative tolerance 1e-9):
  A  one W&B run id across both phases; epochs 1..4 each logged once; phases start at 1 and 3
  B  every cumulative key = previous + this epoch's, including across the resume
  C  per-epoch useful FLOPs, recompute (> 0), tokens, encoder/processor/decoder,
     enc_dec_flop_fraction, flops_per_microbatch equal the planner's exact replay
  D  each checkpoint's saved counters equal W&B's cumulative values at that epoch
  E  repetition_factor = tokens_seen / T_unique_model

Usage: python Runs/test_flops_fsdp/check_flops_fsdp.py <phase1 log>   (from the repo root)
Exit 0 = all PASS.
"""

import glob
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import torch

from walrus.utils.flops import GRANULAR_FLOP_KEYS, stage_totals

TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parents[1]
EXPERIMENT_DIR = "/leonardo_work/ICT26_MHPC_0/sshamsi/logs"
EPOCHS = [1, 2, 3, 4]
PHASE_STARTS = [1, 3]
RTOL = 1e-9
OVERRIDES = [
    line.strip()
    for line in (TEST_DIR / "overrides.txt").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

results = []


def check(group, name, ok, detail=""):
    results.append((group, name, bool(ok), detail))


def close(a, b):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=RTOL, abs_tol=0.0)


def read_wandb_history(run_id):
    """[(set of run ids, [history rows])] per offline run, in start order."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    out = []
    for path in sorted(glob.glob(f"{EXPERIMENT_DIR}/wandb/offline-run-*-{run_id}/run-*.wandb")):
        ds = datastore.DataStore()
        ds.open_for_scan(path)
        rows, ids = [], set()
        while (data := ds.scan_data()) is not None:
            rec = pb.Record()
            rec.ParseFromString(data)
            kind = rec.WhichOneof("record_type")
            if kind == "run":
                ids.add(rec.run.run_id)
            elif kind == "history":
                rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json)
                             for i in rec.history.item})
        out.append((ids, rows))
    return out


def planner(epoch):
    cmd = [sys.executable, "-W", "ignore", str(REPO_DIR / "Tools/flop_planner/count_WALRUS_flops.py"),
           "--nodes", "1", "--gpus-per-node", "4", "--exact", "--json",
           "--start-epoch", str(epoch), "--end-epoch", str(epoch),
           "--override", *OVERRIDES, f"trainer.max_epoch={EPOCHS[-1]}"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_DIR).stdout
    return json.loads(out[out.index("{"):])["job"]


def main(phase1_log):
    m = re.search(r"Using default experiment folder (\S+)", Path(phase1_log).read_text())
    if not m:
        sys.exit(f"no experiment folder in {phase1_log}")
    folder = Path(m.group(1))
    run_id = (folder / "wandb_run_id.txt").read_text().strip()
    print(f"experiment folder  {folder}\nW&B run id         {run_id}\n")

    # --- A: one W&B run across both phases
    phases = read_wandb_history(run_id)
    check("A", "2 offline runs (one per phase)", len(phases) == 2, f"found {len(phases)}")
    check("A", "all carry the folder's run id", all(ids == {run_id} for ids, _ in phases),
          str([sorted(ids) for ids, _ in phases]))
    by_epoch, starts = {}, []
    for _, rows in phases:
        train_rows = [r for r in rows if "epoch_flops" in r]
        if train_rows:
            starts.append(min(int(r["epoch"]) for r in train_rows))
        for r in train_rows:
            by_epoch.setdefault(int(r["epoch"]), []).append(r)
    check("A", "epochs 1..4 each logged exactly once",
          sorted(by_epoch) == EPOCHS and all(len(v) == 1 for v in by_epoch.values()),
          f"logged {sorted((e, len(v)) for e, v in by_epoch.items())}")
    check("A", f"phases start at epochs {PHASE_STARTS}", starts == PHASE_STARTS, f"started at {starts}")
    if sorted(by_epoch) != EPOCHS:
        return report()
    log = {e: by_epoch[e][0] for e in EPOCHS}

    # --- B: cumulative = previous + this epoch, across the resume too
    pairs = [("cumulative_flops", "epoch_flops"), ("cumulative_recompute_flops", "epoch_recompute_flops")]
    pairs += [(f"cumulative_{k}_flops", f"epoch_{k}_flops") for k in GRANULAR_FLOP_KEYS]
    for cumulative, per_epoch in pairs:
        prev, bad = 0.0, []
        for e in EPOCHS:
            if not close(log[e][cumulative], prev + log[e][per_epoch]):
                bad.append(e)
            prev = log[e][cumulative]
        check("B", f"{cumulative}[e] = [e-1] + {per_epoch}[e]", not bad, f"breaks at epochs {bad}" if bad else "")
    toks = [log[e]["tokens_seen"] for e in EPOCHS]
    check("B", "tokens_seen increases every epoch", all(b > a for a, b in zip([0] + toks, toks)), str(toks))

    # --- C: against the planner's exact replay
    plan = {e: planner(e) for e in EPOCHS}
    prev_tok = 0.0
    for e in EPOCHS:
        j, stages = plan[e], stage_totals({k: log[e][f"epoch_{k}_flops"] for k in GRANULAR_FLOP_KEYS})
        rows = [
            ("epoch_flops (useful)", log[e]["epoch_flops"], j["flops_useful"]),
            ("epoch_recompute_flops", log[e]["epoch_recompute_flops"], j["flops_recompute"]),
            ("tokens", log[e]["tokens_seen"] - prev_tok, j["tokens_seen"]),
            ("encoder", stages["encoder"], j["flops_encoder"]),
            ("processor", stages["processor"], j["flops_processor"]),
            ("decoder", stages["decoder"], j["flops_decoder"]),
            ("enc_dec_flop_fraction", log[e]["enc_dec_flop_fraction"], j["forward_enc_dec_fraction"]),
            ("flops_per_microbatch", log[e]["flops_per_microbatch"], j["flops_useful"] / j["microbatches"]),
        ]
        for name, got, want in rows:
            check("C", f"epoch {e}: {name}", close(got, want), f"logged {got:.9e} expected {want:.9e}")
        check("C", f"epoch {e}: recompute > 0", log[e]["epoch_recompute_flops"] > 0)
        prev_tok = log[e]["tokens_seen"]

    # --- D: checkpoints carry the same counters
    for e in EPOCHS:
        meta = folder / "checkpoints" / f"step_{e}" / "metadata.pt"
        saved = torch.load(meta, weights_only=False).get("flop_accountant") if meta.exists() else None
        ok = saved is not None \
            and close(stage_totals(saved["cumulative_granular_flops"])["total"], log[e]["cumulative_flops"]) \
            and close(saved["cumulative_recompute_flops"], log[e]["cumulative_recompute_flops"]) \
            and saved["cumulative_tokens"] == int(log[e]["tokens_seen"])
        check("D", f"step_{e} metadata == W&B cumulative", ok, "missing or no flop_accountant" if saved is None else "")

    # --- E: repetition factor
    unique = plan[EPOCHS[-1]]["unique_model_tokens"]
    for e in EPOCHS:
        check("E", f"epoch {e}: repetition_factor",
              close(log[e]["repetition_factor"], log[e]["tokens_seen"] / unique),
              f"logged {log[e]['repetition_factor']:.6e}, tokens/unique {log[e]['tokens_seen'] / unique:.6e}")

    print(f"{'epoch':>6}{'useful TF':>12}{'planned TF':>12}{'recompute TF':>14}{'tokens_seen':>14}")
    for e in EPOCHS:
        print(f"{e:>6}{log[e]['epoch_flops'] / 1e12:>12.4f}{plan[e]['flops_useful'] / 1e12:>12.4f}"
              f"{log[e]['epoch_recompute_flops'] / 1e12:>14.4f}{log[e]['tokens_seen']:>14,.0f}")
    return report()


def report():
    print()
    for group, name, ok, detail in results:
        print(f"  {group}  {'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail and not ok else ""))
    n_fail = sum(not ok for _, _, ok, _ in results)
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL'} ({len(results)} checks)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main(sys.argv[1])
