# =============================================================================
# tests/test_integration_backends.py
# -----------------------------------------------------------------------------
# Responsible for: Directly testing the detection-source seam (integration/backends.py)
#                  and the pose feed (integration/telemetry.py) — the parts the loop test
#                  only exercises indirectly (and the YOLO path not at all).
# Role in project: Hardens the source-agnostic seam. Both backends must emit a well-formed
#                  brain DetectorOutput with the frame_id / sensor_type / timestamp carried
#                  through from the scripted frame, because those are exactly the join keys
#                  the GeoReferencer relies on downstream. The YOLO path is tested with FAKES
#                  (no torch, no cv2, no weights) so it runs in the brain's light venv.
# Assumptions: SimulatorBackend uses the real DetectorSimulator (cheap, deterministic with a
#              fixed seed). YoloBackend is fed a fake Ultralytics-shaped model + a fake video
#              source, so we assert wiring (right frame index, carried metadata, person
#              filter) without any heavy dependency.
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import BrainConfig
from src.common.contracts import DetectorOutput, SensorType
from src.common.grid import GridSpec
from src.demo.detector_sim import DetectorSimulator
from src.search.terrain import SyntheticTerrain

from integration.backends import DetectorBackend, SimulatorBackend, YoloBackend
from integration.telemetry import VideoFrameSource, build_pose_feed


# --- shared scenario helpers ---------------------------------------------------

def _scenario():
    """Build a grid + scripted pose feed + visibility for the default synthetic scenario."""
    cfg = BrainConfig()
    grid = GridSpec.from_config(cfg)
    terrain = SyntheticTerrain(cfg)
    subject_latlon, frames = build_pose_feed(grid, cfg)
    visibility = terrain.visibility(grid)
    return cfg, grid, subject_latlon, frames, visibility


# --- telemetry: the pose feed --------------------------------------------------

def test_build_pose_feed_returns_subject_and_frames():
    """
    Scenario: build the scripted pose feed for the default synthetic scenario.
    Why it matters: every downstream backend/loop joins detections to poses by frame_id, so
    the feed must be a non-empty list of frames that each carry a pose + sensor + timestamp,
    and a single ground-truth subject (lat, lon).
    """
    _, _, subject_latlon, frames, _ = _scenario()

    assert isinstance(subject_latlon, tuple) and len(subject_latlon) == 2
    assert len(frames) > 0
    first = frames[0]
    # Each scripted frame carries the three things the geo join needs.
    assert first.pose.frame_id
    assert isinstance(first.sensor_type, SensorType)
    assert isinstance(first.timestamp, (int, float))


# --- SimulatorBackend ----------------------------------------------------------

def test_simulator_backend_satisfies_protocol_and_carries_metadata():
    """
    Scenario: wrap a real DetectorSimulator and detect on one overflight frame.
    Why it matters: the loop depends only on the DetectorBackend Protocol, and the geo join
    needs frame_id/sensor/timestamp to match the source pose. We assert structural conformance
    AND that those fields are carried straight from the scripted frame.
    """
    cfg, grid, subject_latlon, frames, visibility = _scenario()
    sim = DetectorSimulator(grid, subject_latlon, visibility, config=cfg, seed=0)
    backend = SimulatorBackend(sim)

    assert isinstance(backend, DetectorBackend)  # structural (Protocol) conformance

    frame = frames[len(frames) // 2]
    out = backend.detect(0, frame)
    assert isinstance(out, DetectorOutput)
    assert out.frame_id == frame.pose.frame_id
    assert out.sensor_type == frame.sensor_type
    assert out.timestamp == frame.timestamp


def test_simulator_backend_eventually_detects_the_subject():
    """
    Scenario: play every frame once, in order, through the simulator backend.
    Why it matters: a detection source that never fires can't drive a locate. Playing the feed
    the way the loop does (one detect() per frame), at least one overflight frame must yield a
    non-empty detection — otherwise the loop's locate would be impossible.
    """
    cfg, grid, subject_latlon, frames, visibility = _scenario()
    sim = DetectorSimulator(grid, subject_latlon, visibility, config=cfg, seed=0)
    backend = SimulatorBackend(sim)

    total_detections = sum(len(backend.detect(i, f).detections) for i, f in enumerate(frames))
    assert total_detections > 0


# --- YoloBackend (fakes: no torch / no cv2) ------------------------------------

class _FakeTensor:
    """Stand-in for an Ultralytics tensor: supports .cpu().numpy() over a NumPy array."""

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeBoxes:
    """Stand-in for result.boxes with the xyxy/conf/cls attributes the adapter reads."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)
        self._n = len(np.asarray(xyxy))

    def __len__(self):
        return self._n


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Minimal Ultralytics-shaped model: a names map + a predict() returning one result."""

    names = {0: "person", 2: "car"}

    def __init__(self, result):
        self._result = result
        self.predict_calls = 0

    def predict(self, frame, conf=0.25, imgsz=960, verbose=False):
        self.predict_calls += 1
        self.last_frame = frame
        return [self._result]


class _FakeVideo:
    """Stand-in for VideoFrameSource: records which frame index was requested."""

    def __init__(self):
        self.requested = []

    def frame(self, index):
        self.requested.append(index)
        return np.zeros((8, 8, 3), dtype=np.uint8)


def test_yolo_backend_reads_right_frame_filters_and_carries_metadata():
    """
    Scenario: a fake model returns one person box + one car box; YoloBackend.detect(7, frame).
    Why it matters: the real-detector path must (a) pull the IMAGE for THIS frame index from
    the video, (b) drop non-person classes, and (c) carry frame_id/sensor/timestamp/model_id
    onto the output so geo can join it. We verify all of that without torch or opencv.
    """
    _, _, _, frames, _ = _scenario()
    scripted = frames[7]

    # person at [10,20,50,140] (xywh -> 10,20,40,120) and a car that must be dropped.
    boxes = _FakeBoxes(
        xyxy=[[10.0, 20.0, 50.0, 140.0], [100.0, 100.0, 160.0, 140.0]],
        conf=[0.9, 0.95],
        cls=[0, 2],
    )
    model = _FakeModel(_FakeResult(boxes))
    video = _FakeVideo()
    backend = YoloBackend(model, video, model_id="fake-yolo", conf=0.3, imgsz=640)

    out = backend.detect(7, scripted)

    # (a) the right video frame was read.
    assert video.requested == [7]
    assert model.predict_calls == 1
    # (b) only the person survived; xyxy was converted to xywh.
    assert isinstance(out, DetectorOutput)
    assert len(out.detections) == 1
    assert out.detections[0].class_name == "person"
    assert out.detections[0].bbox_xywh == (10.0, 20.0, 40.0, 120.0)
    # (c) join metadata carried from the scripted frame + the configured model_id.
    assert out.frame_id == scripted.pose.frame_id
    assert out.sensor_type == scripted.sensor_type
    assert out.timestamp == scripted.timestamp
    assert out.model_id == "fake-yolo"


def test_yolo_backend_empty_result_is_a_clean_non_detection():
    """
    Scenario: the fake model finds nothing on this frame (empty boxes).
    Why it matters: an empty detection list is the meaningful non-detection signal the brain
    lowers probability on — the YOLO path must surface it as a clean empty DetectorOutput, not
    a crash, just like the adapter unit test asserts at the lower level.
    """
    _, _, _, frames, _ = _scenario()
    model = _FakeModel(_FakeResult(_FakeBoxes(xyxy=np.empty((0, 4)), conf=[], cls=[])))
    backend = YoloBackend(model, _FakeVideo(), model_id="fake-yolo")

    out = backend.detect(0, frames[0])
    assert out.detections == []


# --- VideoFrameSource error handling -------------------------------------------

def test_video_frame_source_raises_on_missing_file():
    """
    Scenario: open a video path that does not exist.
    Why it matters: the real-detector path should fail loudly and early (a clear
    FileNotFoundError) rather than silently producing empty frames that look like
    non-detections — that would corrupt the map without any signal something is wrong.
    cv2 is installed in the venv, so this exercises the real OpenCV open path.
    """
    with pytest.raises(FileNotFoundError):
        VideoFrameSource("/no/such/footage_does_not_exist.mp4")
