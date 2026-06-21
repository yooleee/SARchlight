# =============================================================================
# detector_sim.py
# -----------------------------------------------------------------------------
# Responsible for: A SIMULATED detector standing in for the teammate's real one,
#                  so we can test the detector -> geo -> brain workflow end to end
#                  without aerial footage (we have none). Given a planted ground-
#                  truth subject and a CameraPose, it emits a DetectorOutput.
# Role in project: The feasibility harness's input. It is a stand-in for a SEPARATE
#                  track, and it is the ONLY simulated piece — geo and the brain are
#                  the real code, running on real geography.
# Non-tautology (important): the simulator projects ground->pixel with a MORE
#                  accurate model than geo's baseline — it accounts for the gimbal-
#                  pitch forward shift that geo's nadir-centered baseline IGNORES. So
#                  geo back-projecting the pixel lands slightly off, and that error is
#                  real and measurable, not a round-trip that always succeeds.
# Realism: canopy-driven misses (low visibility -> more misses; thermal sees better),
#          box/confidence jitter, and occasional false positives. All seeded.
# =============================================================================

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from src.common.config import DEFAULT_CONFIG, BrainConfig
from src.common.contracts import (
    CameraPose,
    Detection,
    DetectorOutput,
    SensorType,
)
from src.common.grid import GridSpec

# Largest off-nadir angle we model before the flat-ground forward-shift blows up
# (a near-horizon camera is out of scope for the baseline; the demo is near-nadir).
_MAX_OFF_NADIR_DEG = 75.0


class DetectorSimulator:
    """
    Emits a DetectorOutput for a frame, given a planted ground-truth subject.

    Why:
        Lets the geo->brain pipeline run on geographically grounded, realistically noisy
        input. Crucially it is NOT geo's inverse: it adds the pitch forward-shift geo's
        baseline ignores, so the harness measures geo's true localization error.
    """

    def __init__(
        self,
        grid: GridSpec,
        subject_latlon: Tuple[float, float],
        visibility: np.ndarray,
        config: BrainConfig = DEFAULT_CONFIG,
        seed: int = 0,
        color_base_conf: float = 0.62,
        thermal_base_conf: float = 0.86,
        conf_noise: float = 0.04,
        box_jitter_px: float = 5.0,
        false_positive_rate: float = 0.02,
        person_extent_m: float = 1.5,
    ) -> None:
        """
        Args:
            grid: The shared GridSpec.
            subject_latlon: The TRUE subject position (ground truth we plant).
            visibility: (n_rows, n_cols) base visibility (terrain canopy); drives misses.
            config: Tunables (the sensor visibility factor).
            seed: RNG seed — reproducible runs (the demo must be deterministic).
            color_base_conf / thermal_base_conf: mean detector confidence by sensor.
            conf_noise: stddev of confidence noise.
            box_jitter_px: stddev of box-center jitter (the detector isn't pixel-perfect).
            false_positive_rate: per-frame chance of a spurious detection.
            person_extent_m: ground size of a person, for the (cosmetic) box dimensions.
        Why:
            Everything that makes the signal realistic is here and seeded, so the
            feasibility result is reproducible and the noise is honest, not hidden.
        """
        self._grid = grid
        self._subject_e, self._subject_n = grid.latlon_to_local_m(*subject_latlon)
        self.subject_cell = grid.latlon_to_cell(*subject_latlon)
        self._visibility = visibility
        self._cfg = config
        self._rng = np.random.default_rng(seed)
        self._color_conf = color_base_conf
        self._thermal_conf = thermal_base_conf
        self._conf_noise = conf_noise
        self._jitter = box_jitter_px
        self._fp_rate = false_positive_rate
        self._person_m = person_extent_m

    def _subject_to_pixel(self, pose: CameraPose) -> Optional[Tuple[float, float]]:
        """
        Project the ground-truth subject to a pixel, accounting for gimbal pitch.

        Args:
            pose: The camera pose.

        Returns:
            (px, py) if the subject falls within the image, else None (out of frame).

        Why:
            Models the footprint center as nadir PLUS the pitch forward-shift
            (alt * tan(off_nadir)) along the heading — the effect geo's baseline omits.
            Decomposes the subject's offset into along/cross-track (geo's frame) and maps
            to pixels. This is the deliberately-more-accurate half of the round trip.
        """
        off_nadir = min(90.0 - abs(pose.gimbal_pitch_deg), _MAX_OFF_NADIR_DEG)
        forward_shift = pose.altitude_agl_m * math.tan(math.radians(off_nadir))

        theta = math.radians(pose.heading_deg)
        fwd_e, fwd_n = math.sin(theta), math.cos(theta)        # forward unit
        right_e, right_n = math.cos(theta), -math.sin(theta)   # right unit (90° CW)

        nadir_e, nadir_n = self._grid.latlon_to_local_m(*pose.drone_latlon)
        center_e = nadir_e + forward_shift * fwd_e             # footprint center (shifted)
        center_n = nadir_n + forward_shift * fwd_n

        de = self._subject_e - center_e
        dn = self._subject_n - center_n
        along = de * fwd_e + dn * fwd_n        # project onto forward
        cross = de * right_e + dn * right_n    # project onto right

        hfov, vfov = pose.fov_deg
        half_cross = pose.altitude_agl_m * math.tan(math.radians(hfov) / 2.0)
        half_along = pose.altitude_agl_m * math.tan(math.radians(vfov) / 2.0)
        cross_frac = cross / (2.0 * half_cross)
        along_frac = along / (2.0 * half_along)
        if abs(cross_frac) > 0.5 or abs(along_frac) > 0.5:
            return None  # subject is outside this frame's footprint

        width, height = pose.image_size_px
        px = (cross_frac + 0.5) * width
        py = (0.5 - along_frac) * height
        return px, py

    def _person_box(self, px: float, py: float, pose: CameraPose) -> Tuple[float, float, float, float]:
        """
        A jittered pixel box around (px, py) sized for a person at this altitude.

        Args:
            px, py: Subject pixel center.
            pose: The camera pose (for the ground sample distance).

        Returns:
            bbox_xywh (top-left + size) in pixels.

        Why:
            Box SIZE is cosmetic (geo uses the center), but the jittered CENTER is the
            random localization error a real detector has — it tests the brain's kernel.
        """
        width, height = pose.image_size_px
        hfov, vfov = pose.fov_deg
        gsd_x = (2.0 * pose.altitude_agl_m * math.tan(math.radians(hfov) / 2.0)) / width
        gsd_y = (2.0 * pose.altitude_agl_m * math.tan(math.radians(vfov) / 2.0)) / height
        bw = max(4.0, self._person_m / max(gsd_x, 1e-6))
        bh = max(4.0, self._person_m / max(gsd_y, 1e-6))
        jx, jy = self._rng.normal(0.0, self._jitter, size=2)
        return (px + jx - bw / 2.0, py + jy - bh / 2.0, bw, bh)

    def simulate(self, pose: CameraPose, sensor_type: SensorType, timestamp: float) -> DetectorOutput:
        """
        Produce the DetectorOutput for one frame.

        Args:
            pose: The camera pose for this frame.
            sensor_type: COLOR or THERMAL for this frame.
            timestamp: Sim clock time stamped on the output.

        Returns:
            A DetectorOutput (possibly with empty detections = a clean non-detection).

        Why:
            Whether a detection appears EMERGES from geometry (is the subject in frame?)
            and visibility (canopy-driven miss probability, lifted by thermal) — not a
            hand-placed flag. Plus a small false-positive chance. This is what makes the
            subject's map cell emerge from the real projection chain.
        """
        width, height = pose.image_size_px
        detections: List[Detection] = []

        pixel = self._subject_to_pixel(pose)
        if pixel is not None:
            # Detection probability = sensor-modulated visibility at the subject's cell.
            base_vis = float(self._visibility[self.subject_cell[0], self.subject_cell[1]])
            p_detect = min(max(base_vis * self._cfg.visibility_factor_for(sensor_type.value), 0.0), 1.0)
            # Thermal floor: under dense canopy base_vis is low, so even thermal's lift leaves
            # p_detect marginal. The floor (config, default 0.0=off) encodes that body heat
            # penetrates canopy gaps at dusk, so a thermal pass reliably detects a forested
            # subject. Thermal only — color stays canopy-gated (keeps the contrast honest).
            if sensor_type == SensorType.THERMAL:
                p_detect = max(p_detect, self._cfg.thermal_detection_floor)
            if self._rng.random() < p_detect:
                base_conf = self._thermal_conf if sensor_type == SensorType.THERMAL else self._color_conf
                conf = float(np.clip(base_conf + self._rng.normal(0.0, self._conf_noise), 0.0, 1.0))
                detections.append(Detection(self._person_box(*pixel, pose), conf, "person"))

        # Occasional false positive somewhere in frame (tests the brain's FP robustness).
        if self._rng.random() < self._fp_rate:
            fx = float(self._rng.uniform(0, width))
            fy = float(self._rng.uniform(0, height))
            detections.append(Detection((fx - 4, fy - 4, 8, 8), float(self._rng.uniform(0.4, 0.6)), "person"))

        return DetectorOutput(
            frame_id=pose.frame_id,
            timestamp=timestamp,
            detections=detections,
            model_id="detector-sim",
            sensor_type=sensor_type,
        )
