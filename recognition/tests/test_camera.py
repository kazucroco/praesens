"""CameraWorker with a short video file as the camera, so no webcam is needed."""

import time

import cv2
import numpy as np
import pytest
from conftest import ALICE, FakeEmbedder, face

from praesens_face import FaceEngine, Settings
from praesens_face.camera import CameraWorker


def make_video(path, frames=40, fps=20, size=(64, 48)):
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert out.isOpened(), "OpenCV can't write MJPG .avi files here"
    for i in range(frames):
        out.write(np.full((h, w, 3), (i * 5) % 255, np.uint8))
    out.release()
    return path


def wait_for(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def engine():
    emb = FakeEmbedder()
    s = Settings(confirm_frames=1, camera_autostart=False, process_fps=50, presence_seconds=60)
    eng = FaceEngine(s, embedder=emb)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    emb.default = [face(ALICE)]  # every camera frame shows Alice
    yield eng
    eng.stop_camera()


def test_source_parsing(engine):
    engine.settings.camera_source = "0"
    assert CameraWorker(engine).source == 0
    engine.settings.camera_source = "clip.mp4"
    assert CameraWorker(engine).source == "clip.mp4"


def test_video_file_runs_recognition_and_stops_at_the_end(engine, tmp_path):
    engine.settings.camera_source = str(make_video(tmp_path / "clip.avi"))
    events = []
    engine.add_listener(events.append)
    engine.start_camera()
    cam = engine.camera
    assert wait_for(lambda: cam._frames_processed > 0)

    jpeg = cam.latest_jpeg(timeout=2)
    assert jpeg and jpeg.startswith(b"\xff\xd8")  # a JPEG
    snap = cam.snapshot_jpeg()
    assert snap and cv2.imdecode(np.frombuffer(snap, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)

    assert [e.guest_id for e in events] == ["alice"]  # cooldown keeps it to one
    assert engine.presence()["guest_ids"] == ["alice"]

    assert wait_for(lambda: not cam.running)  # end of file stops the worker
    st = cam.stats()
    assert st["running"] is False and st["error"] is None
    assert 0 < st["frames_processed"] <= st["frames_read"] <= 40


def test_start_camera_twice_keeps_one_worker(engine, tmp_path):
    engine.settings.camera_source = str(make_video(tmp_path / "clip.avi", frames=200))
    engine.start_camera()
    first = engine.camera
    engine.start_camera()
    assert engine.camera is first
    engine.stop_camera()
    assert not first.running
    assert engine.presence()["guest_ids"] == []


def test_unopenable_source_reports_an_error(engine, tmp_path):
    engine.settings.camera_source = str(tmp_path / "no-such-file.avi")
    engine.start_camera()
    cam = engine.camera
    assert wait_for(lambda: cam.error is not None)
    assert "cannot open camera source" in cam.stats()["error"]
    assert cam.running  # keeps retrying until stopped
    assert cam.snapshot_jpeg() is None
    assert cam.latest_jpeg(timeout=0.05) is None
    engine.stop_camera()
    assert not cam.running
