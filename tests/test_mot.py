import numpy as np
import pytest

from cmc_tracking.mot import read_detections, read_sequence_info, write_results


def test_read_sequence_info(tmp_path):
    path = tmp_path / "seqinfo.ini"
    path.write_text(
        "[Sequence]\nname=MOT17-05-FRCNN\nimDir=img1\nframeRate=14\nseqLength=837\n"
        "imWidth=640\nimHeight=480\nimExt=.jpg\n"
    )
    info = read_sequence_info(path)
    assert (info.name, info.length, info.fps, info.width, info.height) == (
        "MOT17-05-FRCNN", 837, 14.0, 640, 480,
    )  # fmt: skip


def test_read_detections_groups_by_frame_and_converts_boxes(tmp_path):
    path = tmp_path / "det.txt"
    path.write_text("2,-1,10,20,30,40,0.9\n1,-1,1,2,3,4,0.5\n2,-1,5,5,5,5,0.1\n")
    frames = read_detections(path, length=3)
    np.testing.assert_allclose(frames[0], [[1, 2, 4, 6, 0.5]])
    np.testing.assert_allclose(frames[1], [[10, 20, 40, 60, 0.9], [5, 5, 10, 10, 0.1]])
    assert frames[2].shape == (0, 5)


def test_read_detections_rejects_frames_out_of_range(tmp_path):
    path = tmp_path / "det.txt"
    path.write_text("9,-1,1,2,3,4,0.5\n")
    with pytest.raises(ValueError, match="frame numbers"):
        read_detections(path, length=3)


def test_write_results_uses_motchallenge_format(tmp_path):
    path = tmp_path / "sub" / "result.txt"
    write_results(path, [np.array([[1.0, 2.0, 4.0, 6.0, 7]]), np.empty((0, 5))])
    assert path.read_text() == "1,7,1.00,2.00,3.00,4.00,1,-1,-1,-1\n"
