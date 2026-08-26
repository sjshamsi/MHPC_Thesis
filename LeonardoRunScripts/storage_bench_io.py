#!/usr/bin/env python3
"""Storage micro-benchmark: raw sequential I/O + realistic Well-HDF5 random-window
reads against a single target directory.

Meant to be run once per candidate `well_base_path` location (see storage_bench.sh),
on a compute node rather than the login node, so results reflect what an actual
training job would see. Only needs numpy/h5py - no torch/hydra dependency, so it's
cheap to run standalone.

Two phases:
  1. Sequential write+read of a throwaway file - the standard "how fast is this
     filesystem" number.
  2. Parallel random-window reads out of a real Well-format HDF5 file (staged into
     the target directory), using multiple worker processes each opening their own
     file handle - mirrors MixedWellDataModule's DataLoader (walrus/data/*), which
     draws random (trajectory, time-window) slices from num_workers processes.

Caveat: there's no root access here to actually drop the page cache between write
and read, so `os.posix_fadvise(..., POSIX_FADV_DONTNEED)` is used as a best-effort
hint. If a location's read number looks implausibly fast, rerun with a larger
--size-gb (bigger than the client-side cache) before trusting it.
"""

import argparse
import multiprocessing as mp
import os
import shutil
import time

import h5py
import numpy as np

# n_steps_input + n_steps_output from walrus/configs/data/Leonardo_smallest8_2_3d.yaml
# (the real multi-dataset mixture) - the size of one training sample's time window.
WINDOW = 11
FIELDS = ["t0_fields/concentration", "t1_fields/velocity"]


def sequential_write_read(path: str, size_gb: float, block_mb: int = 64):
    f = os.path.join(path, "_storage_bench_seq.bin")
    block = os.urandom(block_mb * 1024 * 1024)
    n_blocks = max(1, round(size_gb * 1024 / block_mb))
    total_bytes = n_blocks * len(block)

    t0 = time.perf_counter()
    with open(f, "wb", buffering=0) as fh:
        for _ in range(n_blocks):
            fh.write(block)
        fh.flush()
        os.fsync(fh.fileno())
    write_s = time.perf_counter() - t0

    with open(f, "rb") as fh:
        try:
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

    t0 = time.perf_counter()
    with open(f, "rb") as fh:
        while fh.read(block_mb * 1024 * 1024):
            pass
    read_s = time.perf_counter() - t0

    os.remove(f)
    return total_bytes, write_s, read_s


def _read_worker(args):
    file_path, n_reads, seed = args
    rng = np.random.default_rng(seed)
    with h5py.File(file_path, "r") as f:
        dsets = [f[name] for name in FIELDS if name in f]
        n_traj, n_time = dsets[0].shape[0], dsets[0].shape[1]
        total_bytes = 0
        for _ in range(n_reads):
            traj = int(rng.integers(0, n_traj))
            t0 = int(rng.integers(0, n_time - WINDOW))
            for d in dsets:
                total_bytes += d[traj, t0 : t0 + WINDOW].nbytes
    return total_bytes


def random_window_reads(path: str, source: str, workers: int, reads_per_worker: int):
    staged = os.path.join(path, "_storage_bench_sample.hdf5")
    if not os.path.exists(staged) or os.path.getsize(staged) != os.path.getsize(source):
        shutil.copyfile(source, staged)

    jobs = [(staged, reads_per_worker, i) for i in range(workers)]
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.map(_read_worker, jobs)
    wall_s = time.perf_counter() - t0

    os.remove(staged)
    return sum(results), wall_s, workers * reads_per_worker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="candidate storage directory to benchmark")
    ap.add_argument("--source", required=True, help="a real Well HDF5 file to stage and read from")
    ap.add_argument("--size-gb", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reads-per-worker", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.path, exist_ok=True)
    print(f"\n=== {args.path} ===", flush=True)

    total_bytes, write_s, read_s = sequential_write_read(args.path, args.size_gb)
    print(
        f"sequential write : {total_bytes / 1e6 / write_s:8.1f} MB/s  "
        f"({total_bytes / 1e9:.2f} GB in {write_s:.2f}s)"
    )
    print(
        f"sequential read  : {total_bytes / 1e6 / read_s:8.1f} MB/s  "
        f"({total_bytes / 1e9:.2f} GB in {read_s:.2f}s)"
    )

    rbytes, rwall, nreads = random_window_reads(
        args.path, args.source, args.workers, args.reads_per_worker
    )
    print(
        f"random windows   : {rbytes / 1e6 / rwall:8.1f} MB/s  {nreads / rwall:7.1f} samples/s  "
        f"({nreads} reads across {args.workers} workers in {rwall:.2f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
