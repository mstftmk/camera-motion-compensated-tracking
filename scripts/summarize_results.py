"""Turn results/metrics.csv into the README's tables and docs/mot17_results.png.

Usage: python scripts/summarize_results.py --results results
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DOCS = Path(__file__).parents[1] / "docs"
MOVING = {"05", "10", "11", "13"}
DETECTORS = ["FRCNN", "SDP", "DPM"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics = pd.read_csv(args.results / "metrics.csv")
    shutil.copy(args.results / "metrics.csv", DOCS / "mot17_metrics.csv")

    combined = metrics[metrics.sequence == "COMBINED_SEQ"].set_index(["detector", "variant"])
    print("| Detector | Tracker | HOTA | IDF1 | AssA | DetA | MOTA | ID switches |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    labels = {"reference": "Official OC-SORT", "baseline": "This repo", "cmc": "This repo + CMC"}
    for detector in DETECTORS:
        for variant, label in labels.items():
            row = combined.loc[(detector, variant)]
            print(
                f"| {detector} | {label} | {row.HOTA:.2f} | {row.IDF1:.2f} | {row.AssA:.2f} "
                f"| {row.DetA:.2f} | {row.MOTA:.2f} | {int(row.IDSW)} |"
            )

    rows = []
    per_sequence = metrics[metrics.sequence != "COMBINED_SEQ"]
    for (detector, sequence), group in per_sequence.groupby(["detector", "sequence"]):
        table = group.set_index("variant")
        base, cmc = table.loc["baseline"], table.loc["cmc"]
        noise = table[table.index.str.startswith("noise")]
        number = sequence.split("-")[1]
        rows.append(
            {
                "detector": detector,
                "sequence": number,
                "moving": number in MOVING,
                "hota": cmc.HOTA - base.HOTA,
                "idf1": cmc.IDF1 - base.IDF1,
                "idsw": int(cmc.IDSW - base.IDSW),
                "noise_hota": float((noise.HOTA - base.HOTA).abs().max()),
                "noise_idf1": float((noise.IDF1 - base.IDF1).abs().max()),
            }
        )
    gains = pd.DataFrame(rows)

    print("\n| Camera | Cases | Mean HOTA change | Mean IDF1 change | Mean ID switch change | HOTA rose beyond noise |")
    print("|---|---:|---:|---:|---:|---:|")
    for moving, label in ((True, "Moving"), (False, "Static")):
        part = gains[gains.moving == moving]
        print(
            f"| {label} | {len(part)} | {part.hota.mean():+.2f} | {part.idf1.mean():+.2f} "
            f"| {part.idsw.mean():+.1f} | {(part.hota > part.noise_hota).sum()} of {len(part)} |"
        )

    print("\n| Sequence | Camera | Detector | HOTA change | Noise | IDF1 change | ID switch change |")
    print("|---|---|---|---:|---:|---:|---:|")
    ordered = gains.sort_values(["moving", "sequence", "detector"], ascending=[False, True, True])
    for row in ordered.itertuples():
        camera = "Moving" if row.moving else "Static"
        print(
            f"| {row.sequence} | {camera} | {row.detector} | {row.hota:+.2f} "
            f"| {row.noise_hota:.2f} | {row.idf1:+.2f} | {row.idsw:+d} |"
        )

    print("\nDifference from official OC-SORT, HOTA and IDF1, per sequence")
    for detector in DETECTORS:
        part = per_sequence[per_sequence.detector == detector]
        for metric in ("HOTA", "IDF1"):
            pivot = part.pivot(index="sequence", columns="variant", values=metric)
            difference = pivot.baseline - pivot.reference
            print(detector, metric, "max", round(difference.abs().max(), 2), "mean", round(difference.mean(), 2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    sequences = ["05", "10", "11", "13", "02", "04", "09"]
    positions = np.arange(len(sequences))
    for axis, detector in zip(axes, DETECTORS):
        part = gains[gains.detector == detector].set_index("sequence").loc[sequences]
        colors = ["#1f77b4" if number in MOVING else "#9aa5b1" for number in sequences]
        axis.bar(positions, part.hota, color=colors)
        axis.errorbar(
            positions, np.zeros(len(positions)), yerr=part.noise_hota, fmt="none",
            ecolor="black", capsize=4, linewidth=1,
        )  # fmt: skip
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(3.5, color="#d1d9e0", linestyle="--")
        axis.set_xticks(positions, sequences)
        axis.set_title(detector)
        axis.set_xlabel("MOT17 sequence (moving camera | static camera)")
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("HOTA change with CMC")
    fig.tight_layout()
    fig.savefig(DOCS / "mot17_results.png", dpi=120)


if __name__ == "__main__":
    main()
