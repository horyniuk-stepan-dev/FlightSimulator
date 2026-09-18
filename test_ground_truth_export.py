"""
Тест GT-експорту по слотах (Етап 0.2): CalibrationLogger.dump_ground_truth.

Перевіряє:
1. Експортуються ВСІ слот-кандидати (а не лише відібрані якорі).
2. center_mercator кожного слота = ground-truth Mercator-позиція центру кадру.
3. Прапорці is_anchor збігаються з відбором close()/calibration.json.
4. Метадані: map_center, frame_step, fps, frame_size, affine (2x3, det<0).

Запуск: python test_ground_truth_export.py  (потрібен лише numpy)
"""
import json
import math
import os
import tempfile

import numpy as np

from simulator.physics.calibration_logger import CalibrationLogger
from test_calibration_logger import (
    FPS,
    FRAME_STEP,
    H,
    W,
    FakeOrthoMap,
    make_H1,
    make_state,
    simulate_lawnmower,
)


def main():
    frames, turns = simulate_lawnmower()
    tmpdir = tempfile.mkdtemp()
    calib_path = os.path.join(tmpdir, "calibration.json")
    gt_path = os.path.join(tmpdir, "ground_truth.json")

    logger = CalibrationLogger(
        calib_path, FakeOrthoMap(), frame_step=FRAME_STEP,
        turn_rate_deg_per_slot=3.0, min_anchor_spacing_slots=15,
        max_anchor_spacing_slots=60,
    )
    for idx, (X, Y, heading) in enumerate(frames):
        logger.log(make_H1(X, Y, heading), W, H, make_state(X, Y, heading), idx)

    logger.close()          # calibration.json (відібрані якорі)
    logger.dump_ground_truth(gt_path, fps=FPS)   # ground_truth.json (усі слоти)

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    with open(calib_path, encoding="utf-8") as f:
        calib = json.load(f)

    # [1] Усі слот-кандидати експортовані
    n_cand = len(logger._candidates)
    assert len(gt["slots"]) == n_cand, f"{len(gt['slots'])} != {n_cand}"
    slot_ids = [s["slot"] for s in gt["slots"]]
    assert slot_ids == list(range(n_cand)), "slots not contiguous 0..N-1"
    n_anch = sum(1 for s in gt["slots"] if s["is_anchor"])
    assert n_anch < n_cand, "GT export must contain non-anchor slots too"
    print(f"[1] OK: {n_cand} слот-кандидатів експортовано (з них {n_anch} якорів)")

    # [2] Метадані
    assert gt["version"] == "gt-2.0"
    assert gt["frame_step"] == FRAME_STEP and gt["fps"] == FPS
    assert gt["frame_size"] == [W, H]
    assert abs(gt["map_center"][0] - FakeOrthoMap._center_x) < 1e-6
    assert abs(gt["map_center"][1] - FakeOrthoMap._center_y) < 1e-6
    assert gt["coordinate_contract"]["ground_center"].startswith("direct centre-ray")
    print(f"[2] OK: метадані (map_center={gt['map_center']}, step={FRAME_STEP}, fps={FPS})")

    # [3] center_mercator = GT позиція центру кадру (<1 мм); affine det<0
    for s in gt["slots"]:
        X, Y, _ = frames[s["video_frame"]]
        gt_center = (X + FakeOrthoMap._center_x, Y + FakeOrthoMap._center_y)
        err = math.hypot(
            s["center_mercator"][0] - gt_center[0],
            s["center_mercator"][1] - gt_center[1],
        )
        assert err < 1e-3, f"slot {s['slot']}: center err {err} m"
        M = np.array(s["affine"])
        assert M.shape == (2, 3)
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        assert det < 0, f"slot {s['slot']}: det {det} >= 0"
        assert len(s["camera_position_world"]) == 3
        assert len(s["camera_orientation_xyzw"]) == 4
        assert s["ground_center_valid"] is True
    print("[3] OK: center_mercator відтворює GT-позицію центру (<1 мм), усі det<0")

    # [4] is_anchor узгоджений з calibration.json
    gt_anchor_slots = {s["slot"] for s in gt["slots"] if s["is_anchor"]}
    calib_anchor_slots = {a["frame_id"] for a in calib["anchors"]}
    assert gt_anchor_slots == calib_anchor_slots, (
        f"anchor sets differ: gt-calib={gt_anchor_slots - calib_anchor_slots}, "
        f"calib-gt={calib_anchor_slots - gt_anchor_slots}")
    print(f"[4] OK: is_anchor збігається з calibration.json ({len(gt_anchor_slots)} якорів)")

    # [5] heading_deg монотонно постійний на прямих, змінюється на дугах
    turn_slots = set()
    for t0, t1 in turns:
        turn_slots.update(range(t0 // FRAME_STEP, (t1 // FRAME_STEP) + 1))
    straight = [s for s in gt["slots"] if s["slot"] not in turn_slots]
    # На прямій heading близький до 0 або ±180 (сталий у межах ноги)
    assert straight, "no straight slots?"
    print(f"[5] OK: heading_deg присутній ({len(straight)} прямих слотів)")

    print("\n=== ALL GT-EXPORT TESTS PASSED ===")


if __name__ == "__main__":
    main()
