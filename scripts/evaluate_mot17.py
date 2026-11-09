"""Score results from run_mot17.py with TrackEval and write results/metrics.csv.

Usage: python scripts/evaluate_mot17.py --data /path/to/MOT17 --out results
Needs TrackEval: https://github.com/JonathonLuiten/TrackEval
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import trackeval

SEQUENCES = ["02", "04", "05", "09", "10", "11", "13"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--detectors", nargs="+", default=["FRCNN", "SDP", "DPM"])
    args = parser.parse_args()

    rows = []
    for detector in args.detectors:
        names = sorted(p.name for p in (args.out / "trackers").glob(f"{detector}-*"))
        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config.update(
            USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False,
            OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False, DISPLAY_LESS_PROGRESS=True,
        )  # fmt: skip
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update(
            GT_FOLDER=str(args.data / "labels" / "train"),
            TRACKERS_FOLDER=str(args.out / "trackers"),
            OUTPUT_FOLDER=str(args.out / "trackeval"),
            TRACKERS_TO_EVAL=names,
            BENCHMARK="MOT17",
            SPLIT_TO_EVAL="train",
            SEQ_INFO={f"MOT17-{s}-{detector}": None for s in SEQUENCES},
            SKIP_SPLIT_FOL=True,
            TRACKER_SUB_FOLDER="",
            PRINT_CONFIG=False,
        )
        metric_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
        metrics = [
            trackeval.metrics.HOTA(metric_config),
            trackeval.metrics.CLEAR(metric_config),
            trackeval.metrics.Identity(metric_config),
        ]
        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        results, _ = evaluator.evaluate([dataset], metrics)
        per_tracker = results["MotChallenge2DBox"]
        for name in names:
            for sequence, data in per_tracker[name].items():
                hota, clear, identity = (data["pedestrian"][k] for k in ("HOTA", "CLEAR", "Identity"))
                rows.append(
                    {
                        "detector": detector,
                        "variant": name.split("-", 1)[1],
                        "sequence": sequence,
                        "HOTA": 100 * float(hota["HOTA"].mean()),
                        "AssA": 100 * float(hota["AssA"].mean()),
                        "DetA": 100 * float(hota["DetA"].mean()),
                        "IDF1": 100 * float(identity["IDF1"]),
                        "MOTA": 100 * float(clear["MOTA"]),
                        "IDSW": int(clear["IDSW"]),
                        "Frag": int(clear["Frag"]),
                    }
                )
    path = args.out / "metrics.csv"
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
