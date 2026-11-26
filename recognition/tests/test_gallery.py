"""The FAISS gallery on its own."""

import numpy as np
import pytest
from conftest import ALICE, BOB, DIM, STRANGER, near, unit

from praesens_face.gallery import Gallery


@pytest.fixture
def g():
    return Gallery(dim=DIM)


def test_empty_gallery(g):
    assert g.size == 0 and g.guests() == {}
    assert g.search(np.stack([ALICE, BOB])) == [[], []]
    assert "alice" not in g


def test_add_and_search(g):
    assert g.add("alice", ALICE) == 1  # a single vector is accepted too
    assert g.add("bob", np.stack([BOB, near(BOB, 0.9)])) == 2
    assert g.size == 3 and g.guests() == {"alice": 1, "bob": 2}
    assert "alice" in g and "carol" not in g

    [[hit]] = g.search(near(ALICE, 0.7))
    assert hit.guest_id == "alice" and hit.score == pytest.approx(0.7, abs=1e-4)


def test_search_returns_best_photo_per_guest(g):
    g.add("alice", np.stack([near(ALICE, 0.5), ALICE, near(ALICE, 0.6)]))
    g.add("bob", BOB)
    [hits] = g.search(ALICE, top_k=2)
    # Alice appears once (her best photo), then Bob, even though she owns
    # three of the four vectors.
    assert [h.guest_id for h in hits] == ["alice", "bob"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_top_k_larger_than_gallery(g):
    g.add("alice", ALICE)
    [hits] = g.search(ALICE, top_k=5)
    assert [h.guest_id for h in hits] == ["alice"]


def test_replace_and_remove(g):
    g.add("alice", np.stack([ALICE, near(ALICE, 0.9)]))
    assert g.replace("alice", STRANGER) == 1
    assert g.guests() == {"alice": 1}
    assert g.search(STRANGER)[0][0].score == pytest.approx(1.0, abs=1e-4)

    assert g.remove("alice") == 1
    assert g.remove("alice") == 0
    assert g.size == 0 and g.search(STRANGER) == [[]]


def test_clear(g):
    g.add("alice", ALICE)
    g.add("bob", BOB)
    g.clear()
    assert g.size == 0 and g.guests() == {}
    g.add("carol", STRANGER)  # still usable after clear
    assert g.guests() == {"carol": 1}


def test_wrong_dimension_is_rejected(g):
    with pytest.raises(ValueError, match="dim 512"):
        g.add("alice", np.ones(128, np.float32))
    with pytest.raises(ValueError, match="dim 512"):
        g.search(np.ones((2, 128), np.float32))


def test_fingerprint_ignores_insertion_order(g):
    other = Gallery(dim=DIM)
    g.add("alice", np.stack([ALICE, near(ALICE, 0.8)]))
    g.add("bob", BOB)
    other.add("bob", BOB)
    other.add("alice", ALICE)
    other.add("alice", g._index.reconstruct(2).reshape(1, -1))  # the same second photo
    assert g.fingerprint() == other.fingerprint()


def test_fingerprint_changes_with_contents(g):
    empty = g.fingerprint()
    g.add("alice", ALICE)
    with_alice = g.fingerprint()
    assert with_alice != empty
    g.replace("alice", unit(ALICE + 0.1))
    assert g.fingerprint() != with_alice
    g.remove("alice")
    assert g.fingerprint() == empty
    # Same vector under another id is a different gallery.
    g.add("bob", ALICE)
    assert g.fingerprint() != with_alice
