"""Read MOTChallenge files and write results in the same format."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    length: int
    fps: float
    width: int
    height: int


def read_sequence_info(path: str | Path) -> SequenceInfo:
    parser = configparser.ConfigParser()
    parser.read(path)
    section = parser["Sequence"]
    return SequenceInfo(
        name=section["name"],
        length=int(section["seqLength"]),
        fps=float(section["frameRate"]),
        width=int(section["imWidth"]),
        height=int(section["imHeight"]),
    )


def read_detections(path: str | Path, length: int) -> list[np.ndarray]:
    """Return one array per frame with rows of [x1, y1, x2, y2, score]."""
    table = np.loadtxt(path, delimiter=",", ndmin=2)
    frames: list[np.ndarray] = [np.empty((0, 5)) for _ in range(length)]
    if table.size == 0:
        return frames
    order = np.argsort(table[:, 0], kind="stable")
    table = table[order]
    numbers = table[:, 0].astype(int)
    if numbers.min() < 1 or numbers.max() > length:
        raise ValueError(f"frame numbers outside 1..{length} in {path}")
    boxes = np.column_stack(
        [table[:, 2], table[:, 3], table[:, 2] + table[:, 4], table[:, 3] + table[:, 5], table[:, 6]]
    )
    for number in np.unique(numbers):
        frames[number - 1] = boxes[numbers == number]
    return frames


def write_results(path: str | Path, results: list[np.ndarray]) -> None:
    """Write per-frame rows of [x1, y1, x2, y2, id] in MOTChallenge format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for frame, rows in enumerate(results, start=1):
            for x1, y1, x2, y2, track_id in rows:
                output.write(
                    f"{frame},{int(track_id)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n"
                )
