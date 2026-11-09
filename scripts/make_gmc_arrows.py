"""Regenerate docs/gmc_arrows.png: how the image flows when the camera moves.

Applies a known camera motion to a photo and draws the points found by estimate_camera_motion.
Needs matplotlib and scikit-image. Photos come from scikit-image: "coffee" (CC0, Rachel Michetti,
courtesy of Pikolo Espresso Bar) and "chelsea" (CC0, Stefan van der Walt).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage import data  # noqa: E402

from cmc_tracking import estimate_camera_motion  # noqa: E402


def camera(dx=0.0, dy=0.0, angle=0.0, zoom=1.0, center=(300.0, 200.0)) -> np.ndarray:
    radians = np.deg2rad(angle)
    linear = zoom * np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    offset = np.array(center) - linear @ np.array(center) + np.array([dx, dy])
    return np.column_stack([linear, offset])


def move(image: np.ndarray, affine: np.ndarray) -> np.ndarray:
    size = (image.shape[1], image.shape[0])
    return cv2.warpAffine(image, affine, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def mean_error(found: np.ndarray, true: np.ndarray, shape: tuple[int, int]) -> float:
    ys, xs = np.mgrid[40 : shape[0] - 40 : 20, 40 : shape[1] - 40 : 20]
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    a = points @ found[:, :2].T + found[:, 2]
    b = points @ true[:, :2].T + true[:, 2]
    return float(np.linalg.norm(a - b, axis=1).mean())


def draw(axis, image, affine, title, rejected=None) -> None:
    """Draw the estimated motion on a regular grid, to scale, ending at each point's new position."""
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    ys, xs = np.mgrid[30 : image.shape[0] - 10 : 62, 30 : image.shape[1] - 10 : 62]
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    moved = points @ affine[:, :2].T + affine[:, 2]
    delta = moved - points
    style = {"angles": "xy", "scale_units": "xy", "scale": 1, "width": 0.0075, "headwidth": 3.5,
             "headlength": 4, "headaxislength": 3.5, "edgecolor": "black", "linewidth": 0.5}  # fmt: skip
    axis.quiver(points[:, 0], points[:, 1], delta[:, 0], delta[:, 1], color="#2ecc71", **style)
    if rejected is not None:
        start, end = rejected
        axis.quiver(start[:, 0], start[:, 1], *(end - start).T, color="#e74c3c", **style)
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_title(title, fontsize=11)
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> None:
    photo = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2GRAY)
    cases = [
        ("Camera pans", camera(dx=44.0, dy=16.0)),
        ("Camera rotates", camera(angle=8.0)),
        ("Camera zooms in", camera(zoom=1.16)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.95), layout="constrained")
    for axis, (title, true) in zip(axes, cases):
        moved = move(photo, true)
        motion = estimate_camera_motion(photo, moved)
        error = mean_error(motion.affine, true, photo.shape)
        print(f"{title}: reliable {motion.reliable}, inliers {motion.n_inliers}/{motion.n_tracked}, error {error:.2f} px")
        draw(axis, moved, motion.affine, title)

    # A moving object, unlike the camera, disagrees with the fit, so RANSAC rejects its points.
    true = camera(dx=30.0, dy=10.0)
    moved = move(photo, true)
    before = photo.copy()
    cat = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2GRAY)[60:200, 120:310]
    top, left = 245, 70
    height, width = cat.shape
    before[top : top + height, left : left + width] = cat
    moved[top - 16 : top - 16 + height, left - 26 : left - 26 + width] = cat
    motion = estimate_camera_motion(before, moved)
    error = mean_error(motion.affine, true, photo.shape)
    print(f"Moving object: reliable {motion.reliable}, inliers {motion.n_inliers}/{motion.n_tracked}, error {error:.2f} px")
    outliers = ~motion.inliers
    on_object = (
        (motion.source[:, 0] > left) & (motion.source[:, 0] < left + width)
        & (motion.source[:, 1] > top) & (motion.source[:, 1] < top + height)
    )  # fmt: skip
    print("rejected points on the object:", int((outliers & on_object).sum()), "of", int(on_object.sum()))
    picks = np.flatnonzero(outliers & on_object)
    picks = picks[np.random.default_rng(0).permutation(len(picks))[:14]]
    draw(
        axes[3], moved, motion.affine, "Pan, with one object moving on its own",
        rejected=(motion.source[picks], motion.target[picks]),
    )  # fmt: skip

    output = Path(__file__).parents[1] / "docs" / "gmc_arrows.png"
    fig.savefig(output, dpi=110)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
