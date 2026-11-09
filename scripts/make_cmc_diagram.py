"""Regenerate docs/cmc_idea.png, a schematic of why camera motion breaks IoU matching."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import numpy as np  # noqa: E402

from cmc_tracking.geometry import iou_matrix  # noqa: E402


def draw(axis, predicted, detection, title) -> None:
    px, py, pw, ph = predicted
    dx, dy, dw, dh = detection
    axis.add_patch(Rectangle((px, py), pw, ph, fill=False, ec="#d62728", lw=2.5, ls="--", label="Predicted track"))
    axis.add_patch(Rectangle((dx, dy), dw, dh, fill=False, ec="#2ca02c", lw=2.5, label="Detection in new frame"))
    iou = iou_matrix(
        np.array([[px, py, px + pw, py + ph]]), np.array([[dx, dy, dx + dw, dy + dh]])
    )[0, 0]
    axis.set_title(f"{title}\nIoU = {iou:.2f}, " + ("match" if iou >= 0.3 else "no match, ID is lost"), fontsize=11)
    axis.set_xlim(0, 200)
    axis.set_ylim(120, 0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    detection = (60, 20, 40, 80)
    draw(axes[0], (105, 20, 40, 80), detection, "Without CMC, after a 45 px pan")
    draw(axes[1], (62, 21, 40, 80), detection, "With CMC, the prediction moves too")
    axes[0].legend(frameon=False, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(1.05, -0.2))
    fig.tight_layout()
    output = Path(__file__).parents[1] / "docs" / "cmc_idea.png"
    fig.savefig(output, dpi=120)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
