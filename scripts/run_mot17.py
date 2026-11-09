"""Run OC-SORT with and without camera motion compensation on the MOT17 training set.

Usage: python scripts/run_mot17.py --data /path/to/MOT17 --out results

--data must hold labels/train/MOT17-XX-DET (from MOT17Labels.zip) and images/train/MOT17-XX
(from MOT17Det.zip). Camera motion is cached under --out so repeat runs are fast.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from cmc_tracking import OCSort, OCSortConfig, estimate_camera_motion
from cmc_tracking.mot import read_detections, read_sequence_info, write_results

SEQUENCES = ["02", "04", "05", "09", "10", "11", "13"]
# Fixed before seeing any tracking result. DPM scores are not probabilities, so 0 is its
# natural cut. All other tracker settings stay at their defaults.
THRESHOLDS = {"FRCNN": 0.5, "SDP": 0.5, "DPM": 0.0}


def camera_motion(data: Path, cache: Path, sequence: str, detector: str, length: int, frames):
    path = cache / f"MOT17-{sequence}-{detector}.npz"
    if path.exists():
        stored = np.load(path)
        return stored["affines"], stored["reliable"]
    affines = np.tile(np.array([[1.0, 0, 0], [0, 1.0, 0]]), (length, 1, 1))
    reliable = np.ones(length, dtype=bool)
    previous = None
    folder = data / "images" / "train" / f"MOT17-{sequence}" / "img1"
    for index in range(length):
        current = cv2.imread(str(folder / f"{index + 1:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if current is None:
            raise FileNotFoundError(folder / f"{index + 1:06d}.jpg")
        if previous is not None:
            motion = estimate_camera_motion(previous, current, frames[index - 1][:, :4])
            affines[index], reliable[index] = motion.affine, motion.reliable
        previous = current
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, affines=affines, reliable=reliable)
    return affines, reliable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--detectors", nargs="+", default=list(THRESHOLDS))
    args = parser.parse_args()

    summary = {}
    for detector in args.detectors:
        config = OCSortConfig(det_threshold=THRESHOLDS[detector])
        for sequence in SEQUENCES:
            name = f"MOT17-{sequence}-{detector}"
            folder = args.data / "labels" / "train" / name
            info = read_sequence_info(folder / "seqinfo.ini")
            frames = read_detections(folder / "det" / "det.txt", info.length)

            start = time.time()
            affines, reliable = camera_motion(
                args.data, args.out / "motion", sequence, detector, info.length, frames
            )
            for label, use_motion in (("baseline", False), ("cmc", True)):
                tracker = OCSort(config)
                results = [
                    tracker.update(frames[i], affines[i] if use_motion else None)
                    for i in range(info.length)
                ]
                tracker_name = f"{detector}-{label}"
                write_results(
                    args.out / "trackers" / tracker_name / f"{name}.txt", results
                )
            shift = np.hypot(affines[1:, 0, 2], affines[1:, 1, 2])
            summary[name] = {
                "frames": info.length,
                "reliable_share": float(reliable.mean()),
                "median_shift_px": float(np.median(shift)),
                "p95_shift_px": float(np.percentile(shift, 95)),
            }
            print(f"{name}: {summary[name]} in {time.time() - start:.0f}s", flush=True)
    path = args.out / "camera_motion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
