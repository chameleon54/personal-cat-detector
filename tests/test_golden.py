"""
Regression guardrail on your real sample photos (Main_code/Images/).

The first run records tests/golden.json (what the model says today). Review it
once; plan.md says all 8 samples were correct. After that, every push checks:
  * the category (cat/dog/human/other) is unchanged       -> hard fail
  * for cats, the recorded top breed is still in the top 3 -> hard fail
  * for cats, the top breed changed but is still in top 3  -> warning only

After you retrain on purpose and are happy with the new answers:
    UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py
"""
import json
import os
import re
import warnings
from pathlib import Path

import pytest

from conftest import MIME, assert_valid_analysis, sample_images

GOLDEN = Path(__file__).with_name("golden.json")
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

# A file name that starts with one of these must land in that category,
# e.g. cat3.png, dog_beagle.jpg, human-selfie.webp. Other names are not checked.
NAME_HINTS = {"cat": "cat", "kitten": "cat", "dog": "dog", "puppy": "dog",
              "human": "human", "person": "human", "selfie": "human", "people": "human"}

IMAGES = sample_images()


def _hint(name: str):
    m = re.match(r"[a-z]+", name.lower())
    return NAME_HINTS.get(m.group(0)) if m else None


def _summary(body: dict) -> dict:
    out = {"category": body["category"]}
    if body["category"] == "cat":
        out["top3"] = [p["breed"] for p in body["predictions"]]
        out["top1_confidence"] = round(body["predictions"][0]["confidence"], 3)
    return out


@pytest.fixture(scope="module")
def results(api):
    res = {}
    for p in IMAGES:
        r = api.post_file("/analyze", p.name, p.read_bytes(), MIME[p.suffix.lower()])
        res[p.name] = _summary(assert_valid_analysis(r, p.name))
    return res


@pytest.fixture(scope="module")
def golden(results):
    if UPDATE or not GOLDEN.is_file():
        GOLDEN.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        warnings.warn(f"Recorded {GOLDEN.name} with {len(results)} images. "
                      f"Review it, then commit it.")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


pytestmark = pytest.mark.skipif(not IMAGES, reason="no sample images in Main_code/Images/")


@pytest.mark.parametrize("img", [p.name for p in IMAGES])
def test_filename_hint_matches_category(results, img):
    expected = _hint(img)
    if expected is None:
        pytest.skip("file name gives no hint")
    got = results[img]["category"]
    assert got == expected, f"{img}: name says {expected!r}, model says {got!r}"


@pytest.mark.parametrize("img", [p.name for p in IMAGES])
def test_matches_golden(results, golden, img):
    if img not in golden:
        pytest.skip(f"{img} is new; record it with UPDATE_GOLDEN=1")
    old, new = golden[img], results[img]
    assert new["category"] == old["category"], \
        f"{img}: category changed {old['category']!r} -> {new['category']!r}"
    if old["category"] == "cat":
        if new["top3"][0] != old["top3"][0]:
            assert old["top3"][0] in new["top3"], \
                (f"{img}: recorded breed {old['top3'][0]!r} fell out of the top 3 "
                 f"(now {new['top3']})")
            warnings.warn(f"{img}: top breed changed {old['top3'][0]!r} -> "
                          f"{new['top3'][0]!r} (still in top 3)")


def test_golden_has_no_stale_entries(golden):
    names = {p.name for p in IMAGES}
    stale = sorted(set(golden) - names)
    assert not stale, f"golden.json lists missing images {stale}; re-record with UPDATE_GOLDEN=1"
