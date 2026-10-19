"""Measure genuine vs impostor scores on your own photos and suggest thresholds.

Folder layout:

    dataset/
      enrolled/<guest_id>/*.jpg      enrollment photos (one face each)
      probes/<guest_id>/*.jpg        test photos of enrolled guests
      probes/unknown/*.jpg           test photos of people who are NOT enrolled

Probe photos should look like the real camera: frames saved from the webcam
in the lobby/hallway setups you'll test in, not more passport photos.

    python tools/tune_threshold.py dataset --far 0.01 --csv scores.csv

For each threshold it reports TAR (enrolled guests correctly matched) and FAR
(probes matched to the wrong guest, or a non-enrolled person matched to
anyone), then suggests a threshold in the middle of the range that gives the best
TAR while keeping FAR at or under --far.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praesens_face import EnrollmentError, FaceEngine, Settings  # noqa: E402
from praesens_face.embedder import decode_image  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--far", type=float, default=0.01, help="target false accept rate (default 0.01 = 1%%)")
    ap.add_argument("--csv", type=Path, help="write every probe score to this CSV")
    ap.add_argument("--cpu", action="store_true", help="run on CPU")
    args = ap.parse_args()

    settings = Settings.from_env(**({"ctx_id": -1} if args.cpu else {}))
    engine = FaceEngine(settings)

    for gdir in sorted((args.dataset / "enrolled").iterdir()):
        if gdir.is_dir():
            try:
                n = engine.enroll(gdir.name, [p.read_bytes() for p in images(gdir)])
                print(f"enrolled {gdir.name}: {n} photo(s)")
            except EnrollmentError as e:
                print(f"SKIPPED {gdir.name}: {e}")

    rows = []  # (probe_path, true_id or None, best_id, best_score)
    for gdir in sorted((args.dataset / "probes").iterdir()):
        if not gdir.is_dir():
            continue
        true_id = None if gdir.name == "unknown" else gdir.name
        for p in images(gdir):
            img = decode_image(p.read_bytes())
            faces = [f for f in engine.embedder.detect(img) if f.det_score >= settings.min_det_score]
            if not faces:
                print(f"no face: {p}")
                rows.append((str(p), true_id, None, -1.0))
                continue
            face = max(faces, key=lambda f: f.size_px)  # the main subject
            hit = engine.gallery.search(face.embedding, top_k=1)[0]
            best_id, score = (hit[0].guest_id, hit[0].score) if hit else (None, -1.0)
            rows.append((str(p), true_id, best_id, score))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["probe", "true_guest", "best_guest", "score"])
            w.writerows(rows)

    genuine = [r for r in rows if r[1] is not None]
    impostor = [r for r in rows if r[1] is None]
    if not genuine and not impostor:
        sys.exit("no probes found")

    def rates(t: float) -> tuple[float, float]:
        # TAR: enrolled probe matched to the right guest at or above t.
        tar = np.mean([r[3] >= t and r[2] == r[1] for r in genuine]) if genuine else float("nan")
        # False accepts: non-enrolled probe matched to anyone, or an enrolled
        # probe matched to the wrong guest.
        fa = [r[3] >= t for r in impostor] + [r[3] >= t and r[2] != r[1] for r in genuine]
        return float(tar), float(np.mean(fa)) if fa else float("nan")

    gs = [r[3] for r in genuine if r[2] == r[1]]
    im = [r[3] for r in impostor] + [r[3] for r in genuine if r[2] != r[1]]
    if gs:
        print(f"\ngenuine scores:  n={len(gs)} min={min(gs):.3f} median={np.median(gs):.3f}")
    if im:
        print(f"impostor scores: n={len(im)} max={max(im):.3f} median={np.median(im):.3f}")
    if not impostor:
        print("WARNING: no probes/unknown photos, so FAR only counts wrong-guest matches.")

    print("\nthreshold    TAR     FAR")
    for t in np.arange(0.20, 0.71, 0.05):
        tar, far = rates(t)
        print(f"   {t:.2f}    {tar:6.1%}  {far:6.1%}")

    # Among thresholds that meet the FAR target, keep those with the best TAR
    # and take the middle of that range, so the suggestion isn't sitting right
    # on the edge of the highest impostor score.
    grid = [round(float(t), 3) for t in np.arange(0.0, 1.0, 0.005)]
    ok = [(t, *rates(t)) for t in grid]
    ok = [(t, tar) for t, tar, far in ok if far <= args.far]
    if ok:
        best_tar = max(tar for _, tar in ok)
        tied = [t for t, tar in ok if tar == best_tar]
        t = tied[len(tied) // 2]
        tar, far = rates(t)
        print(f"\nthresholds {tied[0]:.3f}-{tied[-1]:.3f} all give TAR {best_tar:.1%} with FAR <= {args.far:.1%}")
        print(f"suggested match_threshold: {t:.3f}  (TAR {tar:.1%}, FAR {far:.1%} on {len(rows)} probes)")
        if len(impostor) < 100:
            print(f"note: only {len(impostor)} non-enrolled probes, so FAR can't be measured "
                  f"below about {1 / max(len(impostor), 1):.0%}; add more 'unknown' photos.")
        print(f"set it live with: curl -X PATCH localhost:8001/faces/config "
              f"-H 'content-type: application/json' -d '{{\"match_threshold\": {t}}}'")
    else:
        print(f"\nno threshold reaches FAR <= {args.far:.1%}")


if __name__ == "__main__":
    main()
