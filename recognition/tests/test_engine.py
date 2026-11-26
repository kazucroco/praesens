"""Fast tests with a fake embedder (no model download needed).

The helpers (FakeEmbedder, ALICE, near, face, ...) are in conftest.py. The HTTP
routes are tested in test_router.py.
"""

import numpy as np
import pytest
from conftest import ALICE, BOB, STRANGER, FakeEmbedder, face, near

from praesens_face import EnrollmentError, FaceEngine, Settings


@pytest.fixture
def fx():
    emb = FakeEmbedder()
    s = Settings(match_threshold=0.5, review_threshold=0.35, confirm_frames=1, cooldown_seconds=60,
                 camera_autostart=False)
    return FaceEngine(s, embedder=emb), emb


def test_enroll_and_match(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.enroll("bob", [emb.scene(2, [face(BOB)])])
    res = eng.identify(emb.scene(3, [face(near(ALICE, 0.7)), face(near(BOB, 0.4)), face(STRANGER)]))
    assert [(r.status, r.guest_id) for r in res][:2] == [("match", "alice"), ("review", "bob")]
    assert res[2].status == "unknown"
    assert res[0].score == pytest.approx(0.7, abs=1e-4)


def test_enrollment_rejects_bad_photos(fx):
    eng, emb = fx
    with pytest.raises(EnrollmentError, match="no clear face"):
        eng.enroll("x", [emb.scene(1, [])])
    with pytest.raises(EnrollmentError, match="exactly one face"):
        eng.enroll("x", [emb.scene(2, [face(ALICE), face(BOB)])])
    with pytest.raises(EnrollmentError, match="no clear face"):
        eng.enroll("x", [emb.scene(3, [face(ALICE, score=0.3)])])
    # One bad photo out of three leaves the gallery untouched.
    with pytest.raises(EnrollmentError, match="photo 2"):
        eng.enroll("x", [emb.scene(4, [face(ALICE)]), emb.scene(5, []), emb.scene(4, [face(ALICE)])])
    assert eng.gallery.size == 0


def test_multi_photo_enrollment_uses_best_photo(fx):
    eng, emb = fx
    a2 = near(ALICE, 0.6)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)]), emb.scene(2, [face(a2)])])
    assert eng.gallery.guests() == {"alice": 2}
    res = eng.identify(emb.scene(3, [face(a2)]))
    assert res[0].guest_id == "alice" and res[0].score == pytest.approx(1.0, abs=1e-4)


def test_threshold_change_is_live(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    probe = emb.scene(2, [face(near(ALICE, 0.45))])
    assert eng.identify(probe)[0].status == "review"
    eng.settings.update({"match_threshold": 0.4})
    assert eng.identify(probe)[0].status == "match"
    with pytest.raises(ValueError):
        eng.settings.update({"review_threshold": 0.9})  # above match_threshold
    assert eng.settings.review_threshold == 0.35  # rolled back


def test_cooldown_and_confirm_frames(fx):
    eng, emb = fx
    eng.settings.update({"confirm_frames": 2})
    events = []
    eng.add_listener(events.append)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    seen = emb.scene(2, [face(near(ALICE, 0.8))])
    empty = emb.scene(3, [])
    eng.process_frame(seen)
    assert events == []  # one frame isn't enough
    eng.process_frame(empty)  # streak broken
    eng.process_frame(seen)
    assert events == []
    eng.process_frame(seen)
    assert len(events) == 1 and events[0].guest_id == "alice"
    for _ in range(5):
        eng.process_frame(seen)
    assert len(events) == 1  # cooldown
    eng.settings.update({"cooldown_seconds": 0})
    eng.process_frame(seen)
    assert len(events) == 2


def test_remove_and_fingerprint(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.enroll("bob", [emb.scene(2, [face(BOB)])])
    before = eng.gallery.fingerprint()
    eng.enroll("carol", [emb.scene(3, [face(STRANGER)])])
    assert eng.gallery.fingerprint() != before
    assert eng.remove("carol") == 1
    assert eng.gallery.fingerprint() == before
    assert eng.identify(emb.scene(4, [face(STRANGER)]))[0].status == "unknown"


def test_rebuild(fx):
    eng, emb = fx
    errors = eng.rebuild([("alice", [emb.scene(1, [face(ALICE)])]), ("bad", [emb.scene(2, [])])])
    assert list(errors) == ["bad"]
    assert eng.gallery.guests() == {"alice": 1}


def test_score_log(fx, tmp_path):
    eng, emb = fx
    eng.settings.score_log_path = str(tmp_path / "scores.csv")
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.process_frame(emb.scene(2, [face(ALICE), face(STRANGER)]))
    lines = (tmp_path / "scores.csv").read_text().splitlines()
    assert lines[0].startswith("timestamp,guest_id,score") and len(lines) == 3


def test_config_file_roundtrip(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    monkeypatch.setenv("PRAESENS_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("PRAESENS_MATCH_THRESHOLD", "0.55")
    s = Settings.from_env()
    assert s.match_threshold == 0.55
    s.update({"match_threshold": 0.6})
    s.save()
    assert Settings.from_env().match_threshold == 0.6


def test_presence(fx):
    eng, emb = fx
    eng.settings.update({"presence_seconds": 0})
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.enroll("bob", [emb.scene(2, [face(BOB)])])
    eng.process_frame(emb.scene(3, [face(near(ALICE, 0.8)), face(near(BOB, 0.4))]))
    assert eng.presence()["guest_ids"] == ["alice"]  # a "review" doesn't count
    eng.process_frame(emb.scene(4, []))
    assert eng.presence()["guest_ids"] == []
    eng.settings.update({"presence_seconds": 60})
    eng.process_frame(emb.scene(3, [face(near(ALICE, 0.8))]))
    eng.process_frame(emb.scene(4, []))
    assert eng.presence()["guest_ids"] == ["alice"]  # held for presence_seconds
    eng.remove("alice")
    eng.process_frame(emb.scene(4, []))
    assert eng.presence()["guest_ids"] == []


def test_enroll_rejects_undecodable_bytes_and_empty_photo_list(fx):
    eng, _ = fx
    with pytest.raises(EnrollmentError, match="photo 1: could not decode image"):
        eng.enroll("x", [b"not an image"])
    with pytest.raises(EnrollmentError, match="at least one photo"):
        eng.enroll("x", [])
    assert eng.gallery.size == 0


def test_small_and_low_confidence_faces_are_ignored(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    assert eng.identify(emb.scene(2, [face(ALICE, size=20)])) == []  # below min_face_px
    assert eng.identify(emb.scene(3, [face(ALICE, score=0.5)])) == []  # below min_det_score
    with pytest.raises(EnrollmentError, match="no clear face"):
        eng.enroll("bob", [emb.scene(4, [face(BOB, size=20)])])


def test_identify_with_empty_gallery_is_unknown(fx):
    eng, emb = fx
    [r] = eng.identify(emb.scene(1, [face(ALICE)]))
    assert (r.status, r.guest_id, r.score) == ("unknown", None, -1.0)


def test_enroll_without_replace_adds_photos(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    assert eng.enroll("alice", [emb.scene(2, [face(near(ALICE, 0.8))])], replace=False) == 2
    assert eng.enroll("alice", [emb.scene(1, [face(ALICE)])]) == 1  # replace is the default


def test_identify_accepts_encoded_bytes(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.png(1, [face(ALICE)])])
    assert eng.identify(emb.png(2, [face(ALICE)]))[0].guest_id == "alice"


def test_review_fires_an_event_but_not_presence(fx):
    eng, emb = fx
    events = []
    eng.add_listener(events.append)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.process_frame(emb.scene(2, [face(near(ALICE, 0.4))]))
    assert [(e.status, e.guest_id) for e in events] == [("review", "alice")]
    assert eng.presence()["guest_ids"] == []


def test_failing_listener_does_not_stop_others(fx):
    eng, emb = fx
    events = []

    def broken(_):
        raise RuntimeError("listener bug")

    eng.add_listener(broken)
    eng.add_listener(events.append)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.process_frame(emb.scene(2, [face(ALICE)]))
    assert len(events) == 1


def test_one_guest_matched_twice_in_a_frame_fires_once(fx):
    eng, emb = fx
    events = []
    eng.add_listener(events.append)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.process_frame(emb.scene(2, [face(near(ALICE, 0.6)), face(near(ALICE, 0.9))]))
    assert len(events) == 1 and events[0].score == pytest.approx(0.9, abs=1e-4)


def test_reenrolling_resets_cooldown(fx):
    eng, emb = fx
    events = []
    eng.add_listener(events.append)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.process_frame(emb.scene(2, [face(ALICE)]))
    eng.process_frame(emb.scene(2, [face(ALICE)]))
    assert len(events) == 1
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.process_frame(emb.scene(2, [face(ALICE)]))
    assert len(events) == 2


def test_stop_camera_clears_presence(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.process_frame(emb.scene(2, [face(ALICE)]))
    assert eng.presence()["guest_ids"] == ["alice"]
    eng.stop_camera()  # no camera was ever started
    assert eng.presence()["guest_ids"] == []


def test_status(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)]), emb.scene(2, [face(near(ALICE, 0.9))])])
    st = eng.status()
    assert st["model_loaded"] is True
    assert (st["guests_enrolled"], st["vectors"]) == (1, 2)
    assert st["camera"] == {"running": False}
    assert st["config"] == eng.settings.tunable()
    assert st["gallery_fingerprint"] == eng.gallery.fingerprint()


def test_numpy_frames_are_used_as_is(fx):
    eng, emb = fx
    frame = emb.scene(1, [face(ALICE)])
    assert isinstance(frame, np.ndarray)
    eng.enroll("alice", [frame])
    assert eng.identify(frame)[0].status == "match"
