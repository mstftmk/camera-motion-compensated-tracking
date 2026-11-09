"""Show camera motion compensation on real MOT17 frames.

Picks the frame where tracking without compensation loses the most people that tracking with
compensation keeps, then draws the camera motion, the predictions and the resulting IDs.
Heads are blurred before drawing. MOT17 frames are licensed CC BY-NC-SA 3.0.

Usage: python scripts/make_real_examples.py --data /path/to/MOT17 --out results/figures
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from cmc_tracking import OCSort, OCSortConfig, estimate_camera_motion  # noqa: E402
from cmc_tracking.geometry import iou_matrix  # noqa: E402
from cmc_tracking.mot import read_detections, read_sequence_info  # noqa: E402

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"]
AFTER = 4


def load_frame(data: Path, sequence: str, frame: int, boxes: np.ndarray) -> np.ndarray:
    path = data / "images" / "train" / f"MOT17-{sequence}" / "img1" / f"{frame + 1:06d}.jpg"
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        top, left = max(int(y1), 0), max(int(x1), 0)
        bottom = min(int(y1 + 0.28 * (y2 - y1)), height)
        right = min(int(x2), width)
        if bottom > top and right > left:
            size = max(int((right - left) * 0.5) | 1, 15)
            image[top:bottom, left:right] = cv2.GaussianBlur(image[top:bottom, left:right], (size, size), 0)
    return image


def read_people(path: Path, length: int) -> list[np.ndarray]:
    table = np.loadtxt(path, delimiter=",", ndmin=2)
    frames = [np.empty((0, 4)) for _ in range(length)]
    for number in np.unique(table[:, 0].astype(int)):
        rows = table[(table[:, 0] == number) & np.isin(table[:, 7], (1, 2, 7))]
        frames[number - 1] = np.column_stack(
            [rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]]
        )
    return frames


def run_sequence(frames, affines, threshold):
    """Run both trackers, recording predictions from the compensated tracker before each update."""
    plain, compensated = OCSort(OCSortConfig(det_threshold=threshold)), OCSort(OCSortConfig(det_threshold=threshold))
    plain_out, cmc_out, predictions = [], [], []
    for index, frame in enumerate(frames):
        snapshot = copy.deepcopy(compensated.tracks)
        rows = []
        for track in snapshot:
            if track.time_since_update == 0:
                track.predict()
                before = track.kalman.box()
                track.warp(affines[index])
                rows.append((track.id, before, track.kalman.box()))
        predictions.append(rows)
        plain_out.append(plain.update(frame))
        cmc_out.append(compensated.update(frame, affines[index]))
    return plain_out, cmc_out, predictions


def identity_at(output: np.ndarray, box: np.ndarray) -> int | None:
    if len(output) == 0:
        return None
    overlap = iou_matrix(box[None], output[:, :4])[0]
    return int(output[np.argmax(overlap), 4]) if overlap.max() > 0.5 else None


def lost_people(index, frames, plain_out, cmc_out, predictions, threshold):
    """People kept by the compensated tracker whose ID changes in the plain tracker at this frame."""
    if index < 1 or index + AFTER >= len(frames):
        return []
    detections = frames[index][frames[index][:, 4] > threshold, :4]
    found = []
    for track_id, before, after in predictions[index]:
        if len(detections) == 0:
            break
        with_cmc = iou_matrix(after[None], detections)[0]
        without = iou_matrix(before[None], detections)[0]
        if with_cmc.max() < 0.3 or without.max() >= 0.3:
            continue
        rows_before = cmc_out[index - 1]
        rows_after = cmc_out[index + AFTER]
        if track_id not in rows_before[:, 4] or track_id not in rows_after[:, 4]:
            continue
        box_before = rows_before[rows_before[:, 4] == track_id, :4][0]
        box_after = rows_after[rows_after[:, 4] == track_id, :4][0]
        id_before = identity_at(plain_out[index - 1], box_before)
        id_after = identity_at(plain_out[index + AFTER], box_after)
        if id_before is not None and id_after is not None and id_before != id_after:
            found.append(
                {"id": track_id, "before": before, "after": after, "det": detections[np.argmax(with_cmc)],
                 "plain_before": id_before, "plain_after": id_after, "box_before": box_before,
                 "box_after": box_after}
            )  # fmt: skip
    return found


def draw_box(axis, box, color, style="-", width=2.0, label=None):
    axis.add_patch(
        Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec=color, ls=style, lw=width, label=label)
    )


def bare(axis, image=None):
    if image is not None:
        axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])


def crop_of(box_a, box_b, shape, margin=70):
    x1 = max(int(min(box_a[0], box_b[0])) - margin, 0)
    y1 = max(int(min(box_a[1], box_b[1])) - margin, 0)
    x2 = min(int(max(box_a[2], box_b[2])) + margin, shape[1])
    y2 = min(int(max(box_a[3], box_b[3])) + margin, shape[0])
    return x1, y1, x2, y2


def mean_best_iou(boxes: list[np.ndarray], detections: np.ndarray) -> float:
    return float(np.mean([iou_matrix(box[None], detections)[0].max() for box in boxes]))


def make_figure(data, out, sequence, detector, index, people, frames, plain_out, cmc_out, predictions, heads):
    folder = data / "images" / "train" / f"MOT17-{sequence}" / "img1"
    current = load_frame(data, sequence, index, heads[index])
    gray_previous = cv2.imread(str(folder / f"{index:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    gray_current = cv2.imread(str(folder / f"{index + 1:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    motion = estimate_camera_motion(gray_previous, gray_current, frames[index - 1][:, :4])
    shift = float(np.hypot(*motion.affine[:, 2]))
    detections = frames[index][frames[index][:, 4] > 0.5, :4]

    fig = plt.figure(figsize=(14, 7.7), layout="constrained")
    grid = fig.add_gridspec(2, 10, height_ratios=[1.3, 1])

    axis = fig.add_subplot(grid[0, :5])
    bare(axis, current)
    for box in frames[index - 1][:, :4]:
        draw_box(axis, box, "white", ":", 0.9)
    keep = np.random.default_rng(0).permutation(len(motion.source))[:170]
    for i in keep:
        start, end = motion.source[i], motion.target[i]
        color = "#2ecc71" if motion.inliers[i] else "#e74c3c"
        axis.annotate(
            "", xy=end, xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "mutation_scale": 9},
        )  # fmt: skip
    axis.set_title(
        f"1. Background points reveal the camera motion: {shift:.0f} px in this frame\n"
        f"Green fits ({motion.n_inliers}), red is rejected ({motion.n_tracked - motion.n_inliers})",
        fontsize=11,
    )

    all_before = mean_best_iou([p[1] for p in predictions[index]], detections)
    all_after = mean_best_iou([p[2] for p in predictions[index]], detections)
    axis = fig.add_subplot(grid[0, 5:])
    bare(axis, current)
    for box in detections:
        draw_box(axis, box, "#2ecc71", "-", 2.2)
    for _, before_box, after_box in predictions[index]:
        draw_box(axis, before_box, "#e74c3c", "--", 1.8)
        draw_box(axis, after_box, "#3498db", "-", 1.5)
    axis.plot([], [], color="#2ecc71", lw=2.2, label="Detection")
    axis.plot([], [], color="#e74c3c", ls="--", lw=1.8, label="Prediction without CMC")
    axis.plot([], [], color="#3498db", lw=1.5, label="Prediction with CMC")
    axis.legend(loc="lower left", fontsize=9, framealpha=0.9)
    axis.set_title(
        f"2. Predictions before matching, every track\nMean best IoU: {all_before:.2f} without CMC, {all_after:.2f} with",
        fontsize=11,
    )

    person = max(people, key=lambda p: p["box_before"][3] - p["box_before"][1])
    x1, y1, x2, y2 = crop_of(person["before"], person["after"], current.shape, margin=45)
    axis = fig.add_subplot(grid[1, :2])
    bare(axis, current[y1:y2, x1:x2])
    offset = np.array([x1, y1, x1, y1])
    draw_box(axis, person["det"] - offset, "#2ecc71", "-", 3)
    draw_box(axis, person["before"] - offset, "#e74c3c", "--", 3)
    draw_box(axis, person["after"] - offset, "#3498db", "-", 2.2)
    iou_before = iou_matrix(person["before"][None], person["det"][None])[0, 0]
    iou_after = iou_matrix(person["after"][None], person["det"][None])[0, 0]
    axis.plot([], [], color="#2ecc71", lw=3, label="Detection")
    axis.plot([], [], color="#e74c3c", ls="--", lw=3, label=f"Prediction without CMC, IoU {iou_before:.2f}")
    axis.plot([], [], color="#3498db", lw=3, label=f"Prediction with CMC, IoU {iou_after:.2f}")
    axis.legend(loc="lower center", fontsize=7.5, framealpha=0.9)
    axis.set_title("3. One person, zoomed", fontsize=10)

    before_image = load_frame(data, sequence, index - 1, heads[index - 1])
    after_image = load_frame(data, sequence, index + AFTER, heads[index + AFTER])
    x1, y1, x2, y2 = crop_of(person["box_before"], person["box_after"], current.shape)
    panels = [
        (grid[1, 2:4], before_image, person["box_before"], f"4. Without CMC, frame {index}", person["plain_before"]),
        (grid[1, 4:6], after_image, person["box_after"], f"Frame {index + AFTER + 1}: new ID", person["plain_after"]),
        (grid[1, 6:8], before_image, person["box_before"], f"With CMC, frame {index}", person["id"]),
        (grid[1, 8:10], after_image, person["box_after"], f"Frame {index + AFTER + 1}: same ID", person["id"]),
    ]
    for spec, image, box, title, shown_id in panels:
        axis = fig.add_subplot(spec)
        bare(axis, image[y1:y2, x1:x2])
        color = COLORS[shown_id % len(COLORS)]
        shifted = box - np.array([x1, y1, x1, y1])
        draw_box(axis, shifted, color, "-", 2.5)
        axis.text(
            shifted[0], shifted[1] - 6, f"ID {shown_id}", color="black", fontsize=11, fontweight="bold",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 2},
        )  # fmt: skip
        axis.set_title(title, fontsize=10)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mot17_example.jpg" if sequence == "10" else out / f"mot17_example_{sequence}.jpg"
    fig.savefig(path, dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)
    return path, shift


def id_changes(reference_out, other_out, gap=AFTER + 1):
    """Count track steps that keep an ID in reference_out but change it in other_out."""
    broken = 0
    for index in range(len(reference_out) - gap):
        now, later = reference_out[index], reference_out[index + gap]
        for row in now:
            match = later[later[:, 4] == row[4]]
            if len(match) == 0:
                continue
            first = identity_at(other_out[index], row[:4])
            second = identity_at(other_out[index + gap], match[0, :4])
            if first is not None and second is not None and first != second:
                broken += 1
    return broken


def read_labeled_people(path: Path, length: int) -> list[np.ndarray]:
    """Per-frame ground truth rows of [x1, y1, x2, y2, person id, visibility]."""
    table = np.loadtxt(path, delimiter=",", ndmin=2)
    frames = [np.empty((0, 6)) for _ in range(length)]
    for number in np.unique(table[:, 0].astype(int)):
        rows = table[(table[:, 0] == number) & np.isin(table[:, 7], (1, 2, 7))]
        frames[number - 1] = np.column_stack(
            [rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5], rows[:, 1], rows[:, 8]]
        )
    return frames


def labeled_person(box: np.ndarray, labels: np.ndarray, min_iou=0.6, min_visibility=0.6) -> int | None:
    """Return the labeled person this box covers well, if that person is clearly visible."""
    if len(labels) == 0:
        return None
    overlap = iou_matrix(box[None], labels[:, :4])[0]
    best = int(np.argmax(overlap))
    if overlap[best] < min_iou or labels[best, 5] < min_visibility:
        return None
    return int(labels[best, 4])


def same_labeled_person(person: dict, index: int, labels: list[np.ndarray]) -> bool:
    """True only if the labels confirm both boxes are the same clearly visible person."""
    first = labeled_person(person["box_before"], labels[index - 1])
    second = labeled_person(person["box_after"], labels[index + AFTER])
    return first is not None and first == second


def cmc_only_loss(index, plain_out, cmc_out):
    """People whose ID stays in plain_out but changes in cmc_out across this frame."""
    if index < 1 or index + AFTER >= len(plain_out):
        return []
    found = []
    for row in plain_out[index - 1]:
        later = plain_out[index + AFTER]
        match = later[later[:, 4] == row[4]]
        if len(match) == 0:
            continue
        first = identity_at(cmc_out[index - 1], row[:4])
        second = identity_at(cmc_out[index + AFTER], match[0, :4])
        if first is not None and second is not None and first != second:
            found.append(
                {"plain": int(row[4]), "cmc_before": first, "cmc_after": second,
                 "box_before": row[:4], "box_after": match[0, :4]}
            )  # fmt: skip
    return found


def id_panel(axis, image, crop, box, title, shown_id) -> None:
    x1, y1, x2, y2 = crop
    bare(axis, image[y1:y2, x1:x2])
    color = COLORS[shown_id % len(COLORS)]
    shifted = box - np.array([x1, y1, x1, y1])
    draw_box(axis, shifted, color, "-", 2.2)
    axis.text(
        shifted[0], shifted[1] - 5, f"ID {shown_id}", color="black", fontsize=9, fontweight="bold",
        bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5},
    )  # fmt: skip
    axis.set_title(title, fontsize=9)


def arrows_panel(axis, data, sequence, index, frames, heads, title_prefix) -> None:
    folder = data / "images" / "train" / f"MOT17-{sequence}" / "img1"
    previous = cv2.imread(str(folder / f"{index:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    current = cv2.imread(str(folder / f"{index + 1:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    motion = estimate_camera_motion(previous, current, frames[index - 1][:, :4])
    bare(axis, load_frame(data, sequence, index, heads[index]))
    for i in np.random.default_rng(0).permutation(len(motion.source))[:110]:
        color = "#2ecc71" if motion.inliers[i] else "#e74c3c"
        axis.annotate(
            "", xy=motion.target[i], xytext=motion.source[i],
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2, "mutation_scale": 7},
        )  # fmt: skip
    shift = float(np.hypot(*motion.affine[:, 2]))
    axis.set_title(f"{title_prefix}, camera moved {shift:.0f} px", fontsize=9)


def gallery(args) -> None:
    """Draw label-confirmed cases: IDs kept only with CMC, and one kept only without it."""
    threshold = 0.5
    wins, losses = [], []
    for sequence in ("13", "11", "10"):
        for detector in ("FRCNN", "SDP"):
            name = f"MOT17-{sequence}-{detector}"
            folder = args.data / "labels" / "train" / name
            info = read_sequence_info(folder / "seqinfo.ini")
            frames = read_detections(folder / "det" / "det.txt", info.length)
            affines = np.load(args.results / "motion" / f"{name}.npz")["affines"]
            plain_out, cmc_out, _ = run_sequence(frames, affines, threshold)
            labels = read_labeled_people(folder / "gt" / "gt.txt", info.length)
            people_boxes = read_people(folder / "gt" / "gt.txt", info.length)
            heads = [np.vstack([people_boxes[i], frames[i][:, :4]]) for i in range(info.length)]
            n_wins = n_losses = 0
            for index in range(1, info.length - AFTER):
                context = (sequence, detector, frames, heads, index)
                for person in cmc_only_loss(index, cmc_out, plain_out):
                    if same_labeled_person(person, index, labels):
                        wins.append((*context, person))
                        n_wins += 1
                for person in cmc_only_loss(index, plain_out, cmc_out):
                    if same_labeled_person(person, index, labels):
                        losses.append((*context, person))
                        n_losses += 1
            print(f"{name}: confirmed kept only with CMC {n_wins}, only without {n_losses}", flush=True)

    def size(case) -> float:
        box = case[-1]["box_before"]
        return box[3] - box[1]

    chosen = []
    for sequence in ("13", "11"):
        options = [case for case in wins if case[0] == sequence]
        if options:
            chosen.append(("win", max(options, key=size)))
    if losses:
        chosen.append(("loss", max(losses, key=size)))

    fig = plt.figure(figsize=(14, 2.75 * len(chosen)), layout="constrained")
    grid = fig.add_gridspec(len(chosen), 5, width_ratios=[2.1, 1, 1, 1, 1])
    for row, (kind, (sequence, detector, frames, heads, index, person)) in enumerate(chosen):
        print(kind, sequence, detector, "frame", index + 1, flush=True)
        label = f"MOT17-{sequence}, {detector}, frame {index + 1}"
        arrows_panel(fig.add_subplot(grid[row, 0]), args.data, sequence, index, frames, heads, label)
        before = load_frame(args.data, sequence, index - 1, heads[index - 1])
        after = load_frame(args.data, sequence, index + AFTER, heads[index + AFTER])
        crop = crop_of(person["box_before"], person["box_after"], before.shape, margin=90)
        if kind == "win":
            # Here the "plain" keys hold the compensated tracker, which kept the ID.
            ids = (person["cmc_before"], person["cmc_after"], person["plain"], person["plain"])
            notes = ("Without CMC", "Without CMC: new ID", "With CMC", "With CMC: same ID")
        else:
            ids = (person["plain"], person["plain"], person["cmc_before"], person["cmc_after"])
            notes = ("Without CMC", "Without CMC: same ID", "With CMC", "With CMC: new ID")
        boxes = (person["box_before"], person["box_after"]) * 2
        images = (before, after, before, after)
        for column, (image, box, note, shown) in enumerate(zip(images, boxes, notes, ids)):
            frame_number = index if column % 2 == 0 else index + AFTER + 1
            axis = fig.add_subplot(grid[row, column + 1])
            id_panel(axis, image, crop, box, f"{note}, frame {frame_number}", shown)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "mot17_gallery.jpg"
    fig.savefig(path, dpi=105, pil_kwargs={"quality": 86})
    plt.close(fig)
    print("wrote", path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    parser.add_argument("--sequences", nargs="+", default=["10"])
    parser.add_argument("--detector", default="FRCNN")
    parser.add_argument("--gallery", action="store_true", help="draw docs/mot17_gallery.jpg instead")
    args = parser.parse_args()
    if args.gallery:
        gallery(args)
        return
    threshold = 0.5

    for sequence in args.sequences:
        name = f"MOT17-{sequence}-{args.detector}"
        folder = args.data / "labels" / "train" / name
        info = read_sequence_info(folder / "seqinfo.ini")
        frames = read_detections(folder / "det" / "det.txt", info.length)
        affines = np.load(args.results / "motion" / f"{name}.npz")["affines"]
        plain_out, cmc_out, predictions = run_sequence(frames, affines, threshold)
        scored = []
        for index in range(len(frames)):
            people = lost_people(index, frames, plain_out, cmc_out, predictions, threshold)
            if people:
                scored.append((len(people), index, people))
        kept_by_cmc = id_changes(cmc_out, plain_out)
        kept_by_plain = id_changes(plain_out, cmc_out)
        print(
            f"{name}: 5-frame ID steps kept only with CMC {kept_by_cmc}, kept only without {kept_by_plain}; "
            f"camera-jump cases {sum(s[0] for s in scored)}",
            flush=True,
        )
        if not scored:
            continue
        scored.sort(key=lambda s: (-s[0], s[1]))
        count, index, people = scored[0]
        people_boxes = read_people(folder / "gt" / "gt.txt", info.length)
        heads = [
            np.vstack([people_boxes[i], frames[i][:, :4]]) for i in range(info.length)
        ]
        path, shift = make_figure(
            args.data, args.out, sequence, args.detector, index, people, frames, plain_out, cmc_out, predictions, heads
        )
        print(f"  frame {index + 1}: {count} people, shift {shift:.1f} px -> {path}", flush=True)


if __name__ == "__main__":
    main()
