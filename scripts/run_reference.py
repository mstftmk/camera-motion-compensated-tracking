"""Run the official OC-SORT code on the same detections, to check this implementation against it.

Usage: python scripts/run_reference.py /path/to/OC_SORT /path/to/MOT17 results
The official code (MIT, github.com/noahcao/OC_SORT) is not part of this repository.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from trackers.ocsort_tracker.ocsort import KalmanBoxTracker, OCSort  # noqa: E402

from cmc_tracking.mot import read_detections, read_sequence_info, write_results  # noqa: E402

data, out = Path(sys.argv[2]), Path(sys.argv[3])
THRESHOLDS = {"FRCNN": 0.5, "SDP": 0.5, "DPM": 0.0}

for detector, threshold in THRESHOLDS.items():
    for sequence in ["02", "04", "05", "09", "10", "11", "13"]:
        name = f"MOT17-{sequence}-{detector}"
        folder = data / "labels" / "train" / name
        info = read_sequence_info(folder / "seqinfo.ini")
        frames = read_detections(folder / "det" / "det.txt", info.length)
        KalmanBoxTracker.count = 0
        tracker = OCSort(
            det_thresh=threshold, max_age=30, min_hits=3, iou_threshold=0.3,
            delta_t=3, asso_func="iou", inertia=0.2, use_byte=False,
        )  # fmt: skip
        size = (info.height, info.width)
        results = [np.asarray(tracker.update(f.copy(), size, size), dtype=float).reshape(-1, 5) for f in frames]
        write_results(out / "trackers" / f"{detector}-reference" / f"{name}.txt", results)
    print("done", detector, flush=True)
