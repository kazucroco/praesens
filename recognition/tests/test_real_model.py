"""End-to-end check with the real buffalo_l model on InsightFace's sample photo.

Downloads the model (~280 MB) on first run. Skipped if insightface isn't installed.
Run with: pytest tests/test_real_model.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

insightface = pytest.importorskip("insightface")

from praesens_face import FaceEngine, Settings  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return FaceEngine(Settings(ctx_id=-1, confirm_frames=1))


def test_enroll_from_crop_then_find_in_group_photo(engine):
    from insightface.data import get_image

    group = get_image("t1")  # six people
    faces = engine.embedder.detect(group)
    assert len(faces) == 6

    # Enroll person 0 from a crop around their face, as staff would upload.
    x1, y1, x2, y2 = faces[0].bbox
    pad = (x2 - x1) // 2
    h, w = group.shape[:2]
    crop = group[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].copy()
    engine.enroll("guest-0", [crop])

    results = engine.identify(group)
    matched = [r for r in results if r.status == "match"]
    assert len(matched) == 1 and matched[0].guest_id == "guest-0"
    others = sorted((r.score for r in results if r.status != "match"), reverse=True)
    print("self score:", round(matched[0].score, 3), "best stranger score:", round(others[0], 3))
    assert others[0] < engine.settings.review_threshold
