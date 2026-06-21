# =============================================================================
# mock_stream.py
# -----------------------------------------------------------------------------
# Responsible for: The scripted FLIGHT PATH for the demo — an ordered list of
#                  CameraPose frames (a lawnmower sweep toward the subject, then an
#                  overflight of it), plus the planted ground-truth subject location.
# Role in project: Drives the real chain now: scripted pose -> detector simulator
#                  (detector_sim.py) -> GeoReferencer -> brain. Replaces the old
#                  direct-Observation mock; the subject's map cell now EMERGES from
#                  the projection instead of being hand-placed. When the real
#                  GeoReferencer/detector exist, this scripted path is the same
#                  CameraPose feed a real flight would provide.
# Assumptions: Near-nadir gimbal; sweep frames are positioned so the subject is NOT
#              in their footprint (clean non-detections); overflight frames sit over
#              the subject so it falls in frame. The sweep is DIRECTION-AGNOSTIC: it
#              flies the band from the LKP toward the subject whichever way that is,
#              so the same builder serves the synthetic (SW) and real-terrain (NW)
#              scenarios. A subject due E/W of the LKP gets a thin sweep (acceptable;
#              both scenarios are diagonal).
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.common.config import BrainConfig
from src.common.contracts import CameraPose, SensorType
from src.common.grid import GridSpec

# Default (SYNTHETIC) subject: ~1.8 km SW of the LKP (demo_scenario §4): moderate
# prior, partial canopy. Used when no explicit offset is passed, so the synthetic
# demo + its regression test are unchanged.
_SUBJECT_OFFSET_ROWS = -26
_SUBJECT_OFFSET_COLS = -23

# REAL-TERRAIN subject: (+16, -16) from the LKP = ~1.13 km NW, in tree cover, at
# ~2.6x the map-mean prior (chosen empirically from the DEM+WorldCover scan — it sits
# in the forested edge of the NW high-prior zone, NEAR but not ON the prior peak, so
# the detections must redirect the map onto it). Exported so the showcase + its
# acceptance test plant the subject at one agreed cell (DRY).
REAL_TERRAIN_SUBJECT_OFFSET: Tuple[int, int] = (16, -16)

_FRAME_DT_S = 9.0

# Camera/flight constants (plausible drone parameters).
_FOV_DEG = (70.0, 40.0)
_IMAGE_PX = (1920, 1080)
_SWEEP_ALT_M = 140.0      # higher sweep => larger footprint => clears the corridor more
_SUBJECT_ALT_M = 90.0     # descend over the subject for detail
_SWEEP_PITCH = -88.0      # near nadir
_COLOR_PITCH = -80.0      # slightly oblique over the subject (exposes geo's baseline error)
_THERMAL_PITCH = -75.0
_COLOR_FRAMES = 6         # loiter long enough that canopy misses still leave >= N detections
_THERMAL_FRAMES = 4


@dataclass(frozen=True)
class ScriptedFrame:
    """
    One frame of the scripted flight: telemetry + which sensor mode + sim clock time.

    Args (fields):
        pose: The CameraPose (telemetry) for this frame.
        sensor_type: COLOR or THERMAL (the camera mode this frame).
        timestamp: Sim-clock time, carried onto the DetectorOutput/Observation.

    Why:
        CameraPose (interfaces §3) carries no timestamp or sensor mode; the demo needs
        both per frame, so this small record pairs them without bending the contract.
    """

    pose: CameraPose
    sensor_type: SensorType
    timestamp: float


def subject_cell(
    grid: GridSpec, cfg: BrainConfig, offset: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """
    The subject's true cell, placed at a (row, col) offset from the LKP.

    Args:
        grid: The shared GridSpec.
        cfg: Brain config (for the LKP).
        offset: (d_row, d_col) cells from the LKP. None -> the synthetic default
            (~1.8 km SW). The real-terrain showcase passes REAL_TERRAIN_SUBJECT_OFFSET.

    Returns:
        (row, col) of the planted subject (ground truth).

    Why:
        The offset is a seam, not a hard constant: the synthetic scenario and the
        real-terrain scenario want the subject in different places (the real prior
        peaks NW, the synthetic SW), so each passes its own offset instead of one
        tuning fighting the other. Placed where the prior is MODERATE and the canopy
        partial, so the update must do real work and the thermal pass earns its
        corroboration.
    """
    d_row, d_col = offset if offset is not None else (_SUBJECT_OFFSET_ROWS, _SUBJECT_OFFSET_COLS)
    lkp_r, lkp_c = grid.latlon_to_cell(*cfg.lkp_latlon)
    return (lkp_r + d_row, lkp_c + d_col)


def _pose(grid: GridSpec, cell, frame_id, heading, alt, pitch) -> CameraPose:
    """A CameraPose whose nadir (drone position) is the center of `cell`."""
    lat, lon = grid.cell_to_latlon(*cell)
    return CameraPose(
        frame_id=frame_id,
        drone_latlon=(lat, lon),
        altitude_agl_m=alt,
        heading_deg=heading,
        gimbal_pitch_deg=pitch,
        fov_deg=_FOV_DEG,
        image_size_px=_IMAGE_PX,
    )


def sweep_pose(grid: GridSpec, cell: Tuple[int, int], frame_id: str, heading: float) -> CameraPose:
    """
    A near-nadir COLOR sweep pose centered on `cell` (sweep altitude/pitch).

    Args:
        grid: The shared GridSpec.
        cell: The (row, col) the drone is over.
        frame_id: Stable frame handle.
        heading: Compass heading (mostly cosmetic at near-nadir; sets the tiny forward shift).

    Returns:
        A CameraPose at the sweep altitude and pitch.

    Why:
        The C1 closed-loop demo flies planner-chosen waypoints; reusing the same sweep
        altitude/pitch constants the scripted path uses keeps the footprint (and thus the
        clearing behaviour) identical between the scripted showcase and the closed loop. (DRY.)
    """
    return _pose(grid, cell, frame_id, heading, _SWEEP_ALT_M, _SWEEP_PITCH)


def loiter_pose(
    grid: GridSpec, cell: Tuple[int, int], frame_id: str, heading: float, sensor_type: SensorType
) -> CameraPose:
    """
    A lower-altitude overflight pose for CONFIRMING a detection (color or thermal pitch).

    Args:
        grid: The shared GridSpec.
        cell: The (row, col) to loiter over (the detection cell).
        frame_id: Stable frame handle.
        heading: Compass heading.
        sensor_type: COLOR or THERMAL (selects the overflight pitch).

    Returns:
        A CameraPose at the descend-for-detail altitude and the sensor's overflight pitch.

    Why:
        On a sweep detection the closed loop loiters to build persistence (and run the thermal
        pass that corroborates under canopy) — the same overflight geometry the scripted
        showcase uses, exposed so the closed loop reuses it rather than re-deriving constants.
    """
    pitch = _THERMAL_PITCH if sensor_type == SensorType.THERMAL else _COLOR_PITCH
    return _pose(grid, cell, frame_id, heading, _SUBJECT_ALT_M, pitch)


def build_scripted_path(
    grid: GridSpec, cfg: BrainConfig, offset: Optional[Tuple[int, int]] = None
) -> Tuple[Tuple[float, float], List[ScriptedFrame]]:
    """
    Build the scripted flight path and the planted subject location.

    Args:
        grid: The shared GridSpec.
        cfg: Brain config (LKP).
        offset: (d_row, d_col) for the subject relative to the LKP. None -> synthetic
            default (SW); the showcase passes REAL_TERRAIN_SUBJECT_OFFSET (NW).

    Returns:
        (subject_latlon, frames): the ground-truth subject position and the ordered
        ScriptedFrames — a lawnmower sweep toward the subject (clean), then a subject
        overflight (color, then thermal).

    Why:
        The path is map-directed: sweep the band between the LKP and the subject
        (clearing it via clean non-detections), then loiter over the subject. The sweep
        flies whichever way the subject lies (SW for synthetic, NW for real terrain), so
        one builder serves both scenarios. The detector simulator decides per frame
        whether the subject is in view, so detections EMERGE from the geometry rather
        than being scripted.
    """
    lkp_r, lkp_c = grid.latlon_to_cell(*cfg.lkp_latlon)
    subject = subject_cell(grid, cfg, offset)
    subject_latlon = grid.cell_to_latlon(*subject)

    frames: List[ScriptedFrame] = []
    t = 0.0
    frame_no = 1

    # --- Sweep: a boustrophedon over the band from the LKP toward the subject. ---
    # Travel along ROWS from just behind the LKP to ~6 cells SHORT of the subject (the
    # buffer keeps the subject out of every sweep footprint -> clean non-detections).
    # sign_r/sign_c point from the LKP toward the subject; treating a 0 delta as +1
    # avoids a zero step (a purely-lateral subject then gets a thin sweep, which is fine
    # for the diagonal scenarios we run). For the SW default this reduces to the old path.
    d_row, d_col = subject[0] - lkp_r, subject[1] - lkp_c
    sign_r = 1 if d_row >= 0 else -1
    row_from = lkp_r - 2 * sign_r            # start just behind the LKP
    row_to = subject[0] - 6 * sign_r         # stop short of the subject's neighborhood
    # Columns span the full band between LKP and subject, with a margin each side.
    col_lo = min(lkp_c, subject[1]) - 3
    col_hi = max(lkp_c, subject[1]) + 3

    serpentine = True
    for row_c in range(row_from, row_to + sign_r, 4 * sign_r):
        cols = list(range(col_lo, col_hi + 1, 5))
        heading = 90.0 if serpentine else 270.0   # east-going vs west-going pass
        if not serpentine:
            cols = cols[::-1]
        serpentine = not serpentine
        for col_c in cols:
            frames.append(ScriptedFrame(
                pose=_pose(grid, (row_c, col_c), f"F{frame_no:04d}", heading, _SWEEP_ALT_M, _SWEEP_PITCH),
                sensor_type=SensorType.COLOR,
                timestamp=t,
            ))
            frame_no += 1
            t += _FRAME_DT_S

    # --- Overflight: loiter over the subject. Color first, then a thermal pass. ---
    # Heading = bearing from the LKP to the subject (0=N, 90=E), so the drone arrives
    # along the approach direction (SW for synthetic, NW for real terrain).
    approach_heading = math.degrees(math.atan2(d_col, d_row)) % 360.0
    for _ in range(_COLOR_FRAMES):    # color frames (canopy -> some misses expected)
        frames.append(ScriptedFrame(
            pose=_pose(grid, subject, f"F{frame_no:04d}", approach_heading, _SUBJECT_ALT_M, _COLOR_PITCH),
            sensor_type=SensorType.COLOR,
            timestamp=t,
        ))
        frame_no += 1
        t += _FRAME_DT_S
    for _ in range(_THERMAL_FRAMES):  # thermal corroboration (sees better under canopy)
        frames.append(ScriptedFrame(
            pose=_pose(grid, subject, f"F{frame_no:04d}", approach_heading, _SUBJECT_ALT_M, _THERMAL_PITCH),
            sensor_type=SensorType.THERMAL,
            timestamp=t,
        ))
        frame_no += 1
        t += _FRAME_DT_S

    return subject_latlon, frames
