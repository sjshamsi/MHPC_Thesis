"""Download The Well datasets to disk for Walrus training.

Wraps the_well's official downloader (resumable curl against its dataset
registry) so runs on this cluster all pull data into a shared, consistent
location. Files land at ``<base-path>/datasets/<dataset>/data/<split>/*.hdf5``,
which is the layout Walrus's ``data.well_base_path`` config expects
(point it at ``<base-path>/datasets/``).
"""

import argparse
import concurrent.futures
import sys

from the_well.data.utils import WELL_DATASETS
from the_well.utils.download import well_download

DEFAULT_BASE_PATH = "/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi"

# The Well datasets actually used by Walrus's main training mixture
# (walrus/configs/data/all_2_3d.yaml), i.e. everything except the
# non-Well/externally-formatted datasets (flowbench, pdebench, pdegym, ...).
DEFAULT_DATASETS = [
    "active_matter",
    "planetswe",
    "acoustic_scattering_maze",
    "acoustic_scattering_inclusions",
    "acoustic_scattering_discontinuous",
    "euler_multi_quadrants_openBC",
    "euler_multi_quadrants_periodicBC",
    "gray_scott_reaction_diffusion",
    "rayleigh_benard",
    "shear_flow",
    "turbulent_radiative_layer_2D",
    "helmholtz_staircase",
    "viscoelastic_instability",
    "supernova_explosion_128",
    "turbulence_gravity_cooling",
    "turbulent_radiative_layer_3D",
    "MHD_64",
    "rayleigh_taylor_instability",
]

ALL_SPLITS = ["train", "valid", "test"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help=f"Directory to hold the 'datasets/' folder (default: {DEFAULT_BASE_PATH})",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=DEFAULT_DATASETS,
        metavar="NAME",
        help="Dataset name(s) to download (default: the sets used by "
        "walrus/configs/data/all_2_3d.yaml). Pass --all for every dataset "
        "the_well ships.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every dataset in the_well's registry instead of --dataset.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=ALL_SPLITS,
        default=ALL_SPLITS,
        help="Split(s) to download (default: all of train, valid, test).",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Only download the first file per split (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--curl-parallel",
        action="store_true",
        help="Let curl fetch a dataset's files concurrently (the_well's --parallel).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of datasets to download concurrently (default: 1, sequential).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print what would be downloaded, without fetching anything.",
    )
    return parser.parse_args()


def download_one(base_path: str, dataset: str, splits, first_only: bool, curl_parallel: bool):
    if set(splits) == set(ALL_SPLITS):
        print(f"[{dataset}] downloading splits={ALL_SPLITS} -> {base_path}/datasets/{dataset}")
        well_download(
            base_path=base_path,
            dataset=dataset,
            split=None,
            first_only=first_only,
            parallel=curl_parallel,
        )
    else:
        for split in splits:
            print(f"[{dataset}] downloading split={split} -> {base_path}/datasets/{dataset}")
            well_download(
                base_path=base_path,
                dataset=dataset,
                split=split,
                first_only=first_only,
                parallel=curl_parallel,
            )
    print(f"[{dataset}] done")


def main():
    args = parse_args()

    datasets = list(WELL_DATASETS) if args.all else args.dataset
    unknown = sorted(set(datasets) - set(WELL_DATASETS))
    if unknown:
        sys.exit(f"Unknown dataset(s): {unknown}. Known datasets: {sorted(WELL_DATASETS)}")

    print(f"Base path: {args.base_path}")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Splits: {args.split}")

    if args.dry_run:
        print("Dry run, exiting without downloading.")
        return

    if args.jobs <= 1:
        for dataset in datasets:
            download_one(args.base_path, dataset, args.split, args.first_only, args.curl_parallel)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    download_one, args.base_path, dataset, args.split, args.first_only, args.curl_parallel
                ): dataset
                for dataset in datasets
            }
            for future in concurrent.futures.as_completed(futures):
                dataset = futures[future]
                future.result()  # re-raise any exception with dataset context below
                print(f"[{dataset}] finished")


if __name__ == "__main__":
    main()
