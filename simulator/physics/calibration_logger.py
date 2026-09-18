"""
CalibrationLogger — генерує calibration.json, сумісний із Topometric Localization.

Принципи синхронізації з DatabaseBuilder:

1. ІНДЕКСАЦІЯ ПО СЛОТАХ БД. DatabaseBuilder семплить кадри як
   range(0, total_frames, frame_step) і кладе кадр відео S*frame_step у слот S
   (див. database_builder.py: prefetch_frames → orig_frame_idx = i // frame_step).
   Кандидат якоря створюється ЛИШЕ на кадрах, кратних frame_step, тож
   frame_id якоря = слот БД, а його матриця відповідає саме тому кадру,
   який DatabaseBuilder покладе у цей слот.

2. 5 ОПОРНИХ ТОЧОК: центр + 4 точки, розкидані по кадру (20%/80%).
   GPS кожної точки береться з ТІЄЇ САМОЇ трансформації, якою рендериться
   кадр — включно з parallax-зміщенням по рельєфу (simulator.terrain.parallax).
   RMSE LSQ-фіту тоді чесно відображає нелінійність, а локалізатор отримує
   повні дані для LOO-QA та інтерполяції.

   ВАЖЛИВО (аудит 2026-08-01): раніше точки бралися з гомографії ПЛОЩИНИ Z=0,
   тоді як рендер зміщує пікселі за висотою рельєфу. Похибка нульова в надирі
   й росте як r·h_rel/altitude — при рельєфі 178 м і висоті 1000 м це ~19 м у
   середині кадру й ~68 м у куті. Усі 5 точок ділили одне й те саме хибне
   припущення, тому власний RMSE якоря лишався ~1e-06 і нічого не показував,
   а pose graph отримував якорі, що суперечать і зображенню, і один одному.

3. ЯКОРІ НА ЗМІНАХ НАПРЯМКУ РУХУ. Кандидат буферизується для КОЖНОГО слота,
   фінальний відбір — у close():
     - межі кожного розвороту (кадр, де напрямок ПОЧИНАЄ змінюватись, і кадр,
       де він СТАБІЛІЗУЄТЬСЯ) + апекс для довгих розворотів;
     - перший та останній слоти запису (щоб clamp-діапазон інтерполятора
       покривав усе відео);
     - проміжні якорі на прямих ділянках (крок ≤ max_anchor_spacing_slots).
   PCHIP-інтерполятор локалізатора отримує вузли саме там, де траєкторія
   ламається — між сусідніми якорями компоненти (rx, ry, sx, sy, angle)
   змінюються монотонно, і інтерполяція не «зрізає кути» розворотів.
"""

import datetime
import json
import math
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from simulator.physics.drone_state import DroneState
from simulator.physics.keyframe_predictor import predict_keyframe_slots
from simulator.terrain.surface_projection import (
    ProjectionGeometryError,
    project_image_pixels_to_surface,
)

if TYPE_CHECKING:  # rasterio потрібен лише реальній мапі, не логеру
    from simulator.terrain.orthophoto_map import OrthophotoMap


class CalibrationLogger:
    def __init__(
        self,
        output_file: str,
        ortho_map: "OrthophotoMap",
        frame_step: int = 30,
        turn_rate_deg_per_slot: float = 3.0,
        min_anchor_spacing_slots: int = 15,
        max_anchor_spacing_slots: int = 60,
        min_speed_m_s: float = 1.0,
        keyframe_criterion: str = "overlap",
        keyframe_max_overlap: float = 0.5,
        keyframe_max_gap_frames: int = 60,
        strict_terrain: bool = True,
        generator_metadata: dict | None = None,
    ):
        """
        Args:
            output_file: шлях до calibration.json
            ortho_map: карта (для local→GPS та зсуву Mercator-центру)
            frame_step: МУСИТЬ збігатися з database.frame_step системи локалізації
            turn_rate_deg_per_slot: поріг швидкості зміни напрямку руху
                (градусів на слот БД), вище якого ділянка вважається розворотом
            min_anchor_spacing_slots: мін. інтервал між ДОДАТКОВИМИ якорями
                на прямих ділянках (якорі розворотів мають пріоритет)
            max_anchor_spacing_slots: макс. інтервал між якорями — довгі прямі
                ділянки добиваються проміжними якорями
            min_speed_m_s: нижче цієї швидкості напрямок руху невизначений —
                використовується yaw
            keyframe_criterion: "overlap" — якорі ставляться ЛИШЕ на слоти, які
                локалізатор залишить keyframe-ами (database.keyframe_criterion
                там мусить бути таким самим); "step" — стара поведінка, якір
                може лягти на будь-який слот
            keyframe_max_overlap: МУСИТЬ збігатися з database.keyframe_max_overlap
            keyframe_max_gap_frames: МУСИТЬ збігатися з database.keyframe_max_gap_frames
        """
        self.output_file = output_file
        self.ortho_map = ortho_map
        self.frame_step = max(1, int(frame_step))
        self.turn_rate_deg = float(turn_rate_deg_per_slot)
        self.min_spacing_slots = max(1, int(min_anchor_spacing_slots))
        self.max_spacing_slots = max(
            self.min_spacing_slots, int(max_anchor_spacing_slots)
        )
        self.min_speed = float(min_speed_m_s)
        self.keyframe_criterion = str(keyframe_criterion)
        self.keyframe_max_overlap = float(keyframe_max_overlap)
        self.keyframe_max_gap_frames = int(keyframe_max_gap_frames)
        self.strict_terrain = bool(strict_terrain)
        self.generator_metadata = dict(generator_metadata or {})

        # Кандидати якорів: по одному на КОЖЕН слот БД з валідною H1
        self._candidates: list[dict] = []
        self._frame_size: tuple[int, int] | None = None
        self._skipped_bad_h = 0

    # ── Внутрішнє ────────────────────────────────────────────────────────────

    def _movement_heading(self, state: DroneState) -> float:
        """Напрямок руху камери (рад): вектор швидкості, fallback — yaw."""
        vx, vy = float(state.velocity[0]), float(state.velocity[1])
        if math.hypot(vx, vy) >= self.min_speed:
            return math.atan2(vy, vx)
        return float(state.yaw)

    @staticmethod
    def _fit_affine_lsq(src_px: np.ndarray, dst_metric: np.ndarray) -> np.ndarray | None:
        """Детермінований LSQ-фіт афінної матриці 2x3 (як у системі локалізації)."""
        n = len(src_px)
        A = np.zeros((2 * n, 6), dtype=np.float64)
        b = np.zeros(2 * n, dtype=np.float64)
        A[0::2, 0] = src_px[:, 0]
        A[0::2, 1] = src_px[:, 1]
        A[0::2, 2] = 1.0
        A[1::2, 3] = src_px[:, 0]
        A[1::2, 4] = src_px[:, 1]
        A[1::2, 5] = 1.0
        b[0::2] = dst_metric[:, 0]
        b[1::2] = dst_metric[:, 1]
        sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 6:
            return None
        return sol.reshape(2, 3)

    @staticmethod
    def select_anchor_indices(
        slots: list[int],
        headings_rad: list[float],
        turn_rate_deg_per_slot: float,
        min_spacing_slots: int,
        max_spacing_slots: int,
    ) -> dict[int, str]:
        """
        Відбирає індекси кандидатів для фінальних якорів.

        Повертає {index → reason}, reason ∈
        {"first", "last", "turn_start", "turn_apex", "turn_end", "straight_fill"}.

        Чиста функція (без стану) — юніт-тестується окремо.
        """
        n = len(slots)
        if n == 0:
            return {}
        if n <= 2:
            return {i: ("first" if i == 0 else "last") for i in range(n)}

        slots_arr = np.asarray(slots, dtype=np.float64)
        h = np.unwrap(np.asarray(headings_rad, dtype=np.float64))
        dh_deg = np.degrees(np.diff(h))
        slot_gaps = np.maximum(np.diff(slots_arr), 1.0)
        rate = np.abs(dh_deg) / slot_gaps  # °/слот між сусідніми кандидатами
        turning = rate >= float(turn_rate_deg_per_slot)

        selected: dict[int, str] = {0: "first", n - 1: "last"}

        # 1) Межі розворотів: кадр, де напрямок починає змінюватись, і кадр,
        #    де він стабілізується. Для довгих розворотів — ще апекс.
        i = 0
        while i < len(turning):
            if not turning[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(turning) and turning[j + 1]:
                j += 1
            # Розворот охоплює кандидатів i .. j+1. Якір на КОЖЕН слот дуги:
            # на дузі камера обертається так швидко, що temporal-матчинг
            # локалізатора між сусідніми слотами неможливий — ці кадри
            # не мають інших обмежень, крім якорів.
            selected.setdefault(i, "turn_start")
            selected.setdefault(j + 1, "turn_end")
            for c in range(i + 1, j + 1):
                selected.setdefault(c, "turn_apex")
            i = j + 1

        # 2) Прямі ділянки: добиваємо проміжними якорями, щоб інтервал
        #    між сусідніми якорями не перевищував max_spacing_slots.
        base = sorted(selected)
        for a, b in zip(base, base[1:]):
            gap = slots_arr[b] - slots_arr[a]
            if gap <= max_spacing_slots:
                continue
            k = int(math.ceil(gap / max_spacing_slots))
            for t in range(1, k):
                target = slots_arr[a] + gap * t / k
                idx = min(range(a + 1, b), key=lambda c: abs(slots_arr[c] - target), default=None)
                if idx is None or idx in selected:
                    continue
                # Не ліпимо fill-якір впритул до вже вибраних
                near = min(abs(slots_arr[idx] - slots_arr[s]) for s in selected)
                if near >= min_spacing_slots:
                    selected[idx] = "straight_fill"

        return selected

    # ── Публічний API ────────────────────────────────────────────────────────

    def log(
        self,
        H1: np.ndarray | None,
        width: int,
        height: int,
        state: DroneState,
        frame_idx: int,
        surface_valid_fraction: float = 1.0,
    ) -> bool:
        """
        Викликається на КОЖЕН кадр відео (frame_idx — індекс кадру у файлі,
        той самий лічильник, що й у VideoWriterSink). Повертає True, якщо
        буферизовано кандидата якоря для слота БД.
        """
        self._frame_size = (int(width), int(height))

        # Тільки кадри, які DatabaseBuilder покладе у слот:
        # range(0, total_frames, frame_step) → слот = frame_idx // frame_step
        if frame_idx % self.frame_step != 0:
            return False
        slot = frame_idx // self.frame_step

        if H1 is None:
            return False
        # 5 опорних точок: центр + 4 розкидані (20% / 80% кадру)
        w, h = float(width), float(height)
        pts_px = np.array(
            [
                [w / 2.0, h / 2.0],
                [0.2 * w, 0.2 * h],
                [0.8 * w, 0.2 * h],
                [0.8 * w, 0.8 * h],
                [0.2 * w, 0.8 * h],
            ],
            dtype=np.float64,
        )
        try:
            projection = project_image_pixels_to_surface(
                H1,
                pts_px,
                self.ortho_map,
                state,
                strict_terrain=self.strict_terrain,
            )
        except ProjectionGeometryError:
            self._skipped_bad_h += 1
            return False
        pts_local = projection.local_xy
        pts_mercator = projection.mercator_xy
        pts_gps = projection.gps
        camera_agl = projection.camera_agl_m

        # LSQ-фіт по всіх 5 точках + чесні метрики залишків
        M = self._fit_affine_lsq(pts_px, pts_mercator)
        if M is None:
            self._skipped_bad_h += 1
            return False

        det = float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
        if det >= 0:
            # px Y↓ → mercator Y↑ мусить давати det<0; інакше щось зламано
            print(
                f"[CalibrationLogger] WARN: candidate at slot {slot} has det={det:.3g} >= 0 "
                f"— skipped (broken projection?)"
            )
            self._skipped_bad_h += 1
            return False

        proj = (M[:, :2] @ pts_px.T).T + M[:, 2]
        errs = np.linalg.norm(proj - pts_mercator, axis=1)

        self._candidates.append(
            {
                "slot": int(slot),
                "video_frame": int(frame_idx),
                "heading": self._movement_heading(state),
                "yaw_deg": math.degrees(state.yaw),
                "alt": float(state.altitude),
                "camera_agl": camera_agl,
                "camera_position": np.asarray(state.position, dtype=float).tolist(),
                "camera_orientation_xyzw": np.asarray(
                    state.quaternion, dtype=float
                ).tolist(),
                "timestamp": float(state.time),
                "ground_center_local": pts_local[0].tolist(),
                "ground_center_mercator": pts_mercator[0].tolist(),
                "ground_center_gps": pts_gps[0],
                "surface_valid_fraction": float(surface_valid_fraction),
                "M": M,
                "rmse": float(np.sqrt(np.mean(errs**2))),
                "median": float(np.median(errs)),
                "max": float(np.max(errs)),
                "pts_px": pts_px.tolist(),
                "pts_gps": pts_gps,
                "pts_mercator": pts_mercator.tolist(),
            }
        )
        return True

    # ── Узгодження з keyframe-селекцією локалізатора ─────────────────────────

    #: Що зберігати першим при колізії двох якорів на одному keyframe.
    #: Менше число = вищий пріоритет. first/last задають clamp-діапазон
    #: інтерполятора, межі розвороту — вузли, де траєкторія ламається.
    _REASON_PRIORITY = {
        "first": 0,
        "last": 0,
        "turn_start": 1,
        "turn_end": 1,
        "turn_apex": 2,
        "straight_fill": 3,
    }

    def _predicted_keyframe_indices(self) -> list[int]:
        """Індекси кандидатів, які локалізатор залишить keyframe-ами."""
        if self.keyframe_criterion != "overlap" or not self._candidates:
            return list(range(len(self._candidates)))
        if not self._frame_size:
            return list(range(len(self._candidates)))
        w, h = self._frame_size
        return predict_keyframe_slots(
            [c["M"] for c in self._candidates],
            w,
            h,
            max_overlap=self.keyframe_max_overlap,
            max_gap_frames=self.keyframe_max_gap_frames,
        )

    def _restrict_to_keyframes(self, selected: dict) -> dict:
        """Пересуває відібрані якорі на найближчі передбачені keyframe-слоти.

        Причина: пропагація локалізатора будує вузли графа лише на слотах із
        фічами. Якір на не-keyframe слоті вона приснеплює до найближчого
        keyframe сама — але при overlap-порозі 0.5 «найближчий» може бути за
        півкадру руху, і зсув мовчки псує калібрацію. Краще зробити снап тут,
        де ще відомо, ЯКИЙ слот був потрібен, і зберегти афінну саме того
        keyframe-а, а не чужу матрицю під чужим номером.

        Колізії (два якорі на один keyframe) розв'язуються тут же: локалізатор
        на такому дублікаті зупиняє пропагацію з помилкою.
        """
        if self.keyframe_criterion != "overlap" or not selected:
            return selected

        kf = self._predicted_keyframe_indices()
        if not kf:
            return selected

        kf_arr = np.asarray(kf, dtype=np.int64)
        kf_set = set(kf)
        moved = 0
        dropped = 0
        out: dict[int, str] = {}
        for idx in sorted(selected, key=lambda i: (self._REASON_PRIORITY.get(selected[i], 99), i)):
            reason = selected[idx]
            if idx in kf_set:
                target = idx
            else:
                target = int(kf_arr[np.argmin(np.abs(kf_arr - idx))])
                moved += 1
            if target in out:
                # Колізію розв'язує пріоритет причини, а не порядок слотів:
                # межі запису й межі розвороту інформативніші за апекс і за
                # добивання прямих ділянок.
                print(
                    f"[CalibrationLogger] WARN: anchor '{reason}' (slot "
                    f"{self._candidates[idx]['slot']}) collapses onto keyframe slot "
                    f"{self._candidates[target]['slot']}, already taken by "
                    f"'{out[target]}' — dropped"
                )
                dropped += 1
                continue
            out[target] = reason if target == idx else f"{reason}+kf-snap"

        if dropped:
            print(
                f"[CalibrationLogger] {dropped} anchor(s) dropped as duplicates. Причина —"
                f" keyframe-и рідші за якорі розворотів; якщо це вадить точності,"
                f" підніміть keyframe_max_overlap в ОБОХ проєктах (менш агресивний"
                f" відбір → щільніші keyframe-и)."
            )

        if moved:
            print(
                f"[CalibrationLogger] {moved}/{len(selected)} anchors moved onto predicted "
                f"keyframes (overlap<={self.keyframe_max_overlap}, "
                f"max_gap={self.keyframe_max_gap_frames}); {len(kf)}/{len(self._candidates)} "
                f"slots are keyframes"
            )
        return out

    def dump_ground_truth(self, path: str, fps: float = 30.0) -> None:
        """Скидає GT ПО КОЖНОМУ слоту (Етап 0.2) для validate_vs_telemetry.py.

        На відміну від close() (лише відібрані якорі), тут експортуються ВСІ
        кандидати — по одному на кожен слот БД з валідною H1. Це дає валідатору
        точну GT-афінну кожного слота (центр + кут + масштаб), а не тільки
        центр із телеметрії, і позначку, які слоти стали фінальними якорями
        (для розрізу «±k від якоря»). Read-only, не впливає на calibration.json.
        """
        if not self._candidates:
            print("[CalibrationLogger] No candidates — ground_truth.json not written.")
            return

        selected = self._restrict_to_keyframes(
            self.select_anchor_indices(
                slots=[c["slot"] for c in self._candidates],
                headings_rad=[c["heading"] for c in self._candidates],
                turn_rate_deg_per_slot=self.turn_rate_deg,
                min_spacing_slots=self.min_spacing_slots,
                max_spacing_slots=self.max_spacing_slots,
            )
        )

        if self._frame_size:
            cx, cy = self._frame_size[0] / 2.0, self._frame_size[1] / 2.0
        else:
            cx, cy = 0.0, 0.0

        slots = []
        for idx, c in enumerate(self._candidates):
            M = np.asarray(c["M"], dtype=np.float64)
            # The direct centre ray is ground truth. The affine centre is a
            # lossy compatibility approximation and is exported separately.
            center = np.asarray(c["ground_center_mercator"], dtype=np.float64)
            affine_center = M[:, :2] @ np.array([cx, cy]) + M[:, 2]
            sx = float(np.hypot(M[0, 0], M[1, 0]))
            sy = float(np.hypot(M[0, 1], M[1, 1]))
            angle_deg = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
            slots.append(
                {
                    "slot": int(c["slot"]),
                    "video_frame": int(c["video_frame"]),
                    "affine": M.tolist(),
                    "center_mercator": [float(center[0]), float(center[1])],
                    "affine_center_mercator": [
                        float(affine_center[0]), float(affine_center[1])
                    ],
                    "ground_center_local": c["ground_center_local"],
                    "ground_center_gps": c["ground_center_gps"],
                    "ground_center_valid": True,
                    "camera_position_world": c["camera_position"],
                    "camera_orientation_xyzw": c["camera_orientation_xyzw"],
                    "camera_agl": float(c["camera_agl"]),
                    "timestamp": float(c["timestamp"]),
                    "surface_valid_fraction": float(c["surface_valid_fraction"]),
                    "sx": sx,
                    "sy": sy,
                    "angle_deg": angle_deg,
                    "heading_deg": float(math.degrees(c["heading"])),
                    "yaw_deg": float(c["yaw_deg"]),
                    "alt": float(c["alt"]),
                    "rmse_m": float(c["rmse"]),
                    "is_anchor": idx in selected,
                    "anchor_reason": selected.get(idx),
                }
            )

        data = {
            "version": "gt-2.0",
            "projection": {"mode": "WEB_MERCATOR", "reference_gps": None},
            "frame_size": list(self._frame_size) if self._frame_size else None,
            "frame_step": int(self.frame_step),
            "fps": float(fps),
            "map_center": [
                float(self.ortho_map._center_x),
                float(self.ortho_map._center_y),
            ],
            "coordinate_contract": {
                "world_xy": "local ground-plane metres, Mercator scale corrected at map centre",
                "world_z": "height above the simulator base plane",
                "ground_center": "direct centre-ray surface intersection",
                "affine": "five-point least-squares compatibility approximation",
            },
            "generator": self.generator_metadata,
            "slots": slots,
        }

        try:
            payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            directory = os.path.dirname(os.path.abspath(path)) or "."
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_gt_", suffix=".part")
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            n_anch = sum(1 for sl in slots if sl["is_anchor"])
            print(
                f"[CalibrationLogger] Saved ground truth: {len(slots)} slot candidates "
                f"({n_anch} marked as anchors) to {path}"
            )
        except Exception as e:
            print(f"[CalibrationLogger] Failed to save ground truth: {e}")

    def close(self) -> None:
        """Відбирає якорі на змінах напрямку та атомарно зберігає
        calibration.json (формат v2.3 системи локалізації)."""
        if not self._candidates:
            print("[CalibrationLogger] No anchor candidates recorded, skipping file creation.")
            return

        selected = self._restrict_to_keyframes(
            self.select_anchor_indices(
                slots=[c["slot"] for c in self._candidates],
                headings_rad=[c["heading"] for c in self._candidates],
                turn_rate_deg_per_slot=self.turn_rate_deg,
                min_spacing_slots=self.min_spacing_slots,
                max_spacing_slots=self.max_spacing_slots,
            )
        )

        now_iso = datetime.datetime.now().isoformat()
        anchors = []
        for idx in sorted(selected):
            c = self._candidates[idx]
            reason = selected[idx]
            anchors.append(
                {
                    "frame_id": c["slot"],
                    "affine_matrix": c["M"].tolist(),
                    "qa_data": {
                        "rmse_m": c["rmse"],
                        "median_err_m": c["median"],
                        "max_err_m": c["max"],
                        "inliers_count": 5,
                        "points_2d": c["pts_px"],
                        "points_gps": c["pts_gps"],
                        "points_metric": c["pts_mercator"],
                        "transform_type": "simulator_ground_truth_lsq5",
                        "projection_mode": "WEB_MERCATOR",
                        "created_at": now_iso,
                        "updated_at": now_iso,
                        "notes": (
                            f"Simulator GT anchor [{reason}] | "
                            f"video_frame={c['video_frame']}, db_slot={c['slot']}, "
                            f"frame_step={self.frame_step}, "
                            f"yaw={c['yaw_deg']:.1f}deg, alt={c['alt']:.0f}m"
                        ),
                        "quality_flag": "normal",
                    },
                }
            )

        data = {
            # 2.4: додано блок keyframe_selection. Стратегія відбору кадрів
            # тепер частина контракту між проєктами — при розбіжності порогів
            # якорі лягають на слоти без фічей, і локалізатор або зсуває їх
            # снапом, або зупиняє пропагацію на колізії.
            "version": "2.4",
            # "mode" — ключ, який читає CoordinateConverter.from_metadata
            "projection": {"mode": "WEB_MERCATOR", "reference_gps": None},
            "frame_size": list(self._frame_size) if self._frame_size else None,
            "keyframe_selection": {
                "criterion": self.keyframe_criterion,
                "max_overlap": self.keyframe_max_overlap,
                "max_gap_frames": self.keyframe_max_gap_frames,
                "frame_step": int(self.frame_step),
            },
            "generator": self.generator_metadata,
            "anchors": anchors,
        }

        try:
            payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            directory = os.path.dirname(os.path.abspath(self.output_file)) or "."
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_calib_", suffix=".part")
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.output_file)
            reasons = {selected[i] for i in selected}
            print(
                f"[CalibrationLogger] Saved {len(anchors)} anchors "
                f"(from {len(self._candidates)} slot candidates) to {self.output_file}\n"
                f"  slots: {[a['frame_id'] for a in anchors]}\n"
                f"  reasons used: {sorted(reasons)}; skipped bad-H candidates: {self._skipped_bad_h}"
            )
        except Exception as e:
            print(f"[CalibrationLogger] Failed to save calibration: {e}")
