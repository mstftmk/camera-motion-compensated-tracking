"""Measure how much tracking scores move when the camera input is meaningless noise.

The tracker is sensitive: a tiny change in one frame can change a later match and everything
after it. Random shifts of about 0.15 px contain no real camera motion, so any score change here
is noise. A claimed gain from camera motion compensation should be larger than this.

Usage: python scripts/run_noise_control.py --data /path/to/MOT17 --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cmc_tracking import OCSort, OCSortConfig
from cmc_tracking.mot import read_detections, read_sequence_info, write_results

SEQUENCES = ["02", "04", "05", "09", "10", "11", "13"]
THRESHOLDS = {"FRCNN": 0.5, "SDP": 0.5, "DPM": 0.0}
SEEDS = (0, 1, 2)
SHIFT_STD_PX = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    for detector, threshold in THRESHOLDS.items():
        for sequence in SEQUENCES:
            name = f"MOT17-{sequence}-{detector}"
            folder = args.data / "labels" / "train" / name
            info = read_sequence_info(folder / "seqinfo.ini")
            frames = read_detections(folder / "det" / "det.txt", info.length)
            for seed in SEEDS:
                rng = np.random.default_rng([seed, int(sequence)])
                tracker = OCSort(OCSortConfig(det_threshold=threshold))
                results = []
                for frame in frames:
                    affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])
                    affine[:, 2] = rng.normal(0.0, SHIFT_STD_PX, 2)
                    results.append(tracker.update(frame, affine))
                write_results(args.out / "trackers" / f"{detector}-noise{seed}" / f"{name}.txt", results)
        print("done", detector, flush=True)


if __name__ == "__main__":
    main()
