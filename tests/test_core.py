import numpy as np

from hoopid.evaluate import evaluate, idf1
from hoopid.ocr import aggregate
from hoopid.tracker import iou_matrix


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], float)
    m = iou_matrix(a, b)[0]
    assert abs(m[0] - 1) < 1e-6 and abs(m[1] - 25 / 175) < 1e-6 and m[2] == 0


def test_idf1_penalises_false_ids_on_other_people():
    pairs = [("a", "P1")] * 10
    assert idf1(pairs) == 1.0
    assert idf1(pairs, extra_pred=10) < 0.7


def test_ocr_needs_several_agreeing_frames():
    assert aggregate([(1, "7", 0.95)])["confirmed"] is None
    assert aggregate([(1, "7", 0.95), (2, "7", 0.9), (3, "7", 0.85)])["confirmed"] == "7"
    assert aggregate([(1, "7", 0.95), (2, "7", 0.5), (3, "7", 0.4)])["confirmed"] is None


def test_unknown_is_not_counted_as_wrong():
    gt = {1: "x", 2: "x", 3: "OTHER"}
    pred = {1: {"frame": 0, "track_id": 1, "pid": "P1", "pid_online": None},
            2: {"frame": 1, "track_id": 1, "pid": None, "pid_online": None},
            3: {"frame": 1, "track_id": 2, "pid": "P1", "pid_online": None}}
    r = evaluate(pred, gt, {"P1": "x"}, (0, 10))
    assert r["correct_confirmed"] == 1 and r["wrong_on_other_people"] == 1 and r["unknown_share"] == 0.5
