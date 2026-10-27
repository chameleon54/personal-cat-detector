"""
Shared fixtures for the guardrail test suite.

Two ways to run:
  * default: the FastAPI app is imported in-process (fastapi.testclient), so
    no server has to be running:            python -m pytest tests
  * live:    test an already-running server:
             GUARDRAILS_BASE_URL=http://127.0.0.1:8000 python -m pytest tests
"""
from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = next((ROOT / n for n in ("Main_code", "main_code") if (ROOT / n).is_dir()), None)
IMAGES_DIR = BACKEND / "Images" if BACKEND else None

# ---------------------------------------------------------------- contract --
CATEGORIES = {"cat", "dog", "human", "other"}
MESSAGES = {"dog": "It's a dog", "human": "It's a human", "other": "Not a cat"}
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx",
}
# If you add a label such as "Mixed / uncertain breed" (plan 5.4 #5), list it here.
EXTRA_BREED_LABELS: set[str] = set()
MAX_PREDICTIONS = 3                       # plan: top-3
MAX_LATENCY_S = float(os.environ.get("GUARDRAILS_MAX_LATENCY_S", "5"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp"}


@dataclass
class Resp:
    status: int
    body: dict | list | None
    text: str


class ApiClient:
    """Same interface whether we talk to TestClient or a live server."""

    def __init__(self, base_url: str | None):
        self.base_url = base_url
        if base_url:
            import requests
            self._s = requests.Session()
        else:
            if BACKEND is None:
                pytest.exit("No Main_code/ folder found next to tests/", returncode=2)
            # main.py loads models with relative paths, so run from its folder.
            os.chdir(BACKEND)
            sys.path.insert(0, str(BACKEND))
            from fastapi.testclient import TestClient
            import main  # noqa: E402  (the user's FastAPI app)
            # raise_server_exceptions=False: a crash shows up as a 500 the
            # tests can report clearly, the same as against a live server.
            self._tc = TestClient(main.app, raise_server_exceptions=False)
            self._tc.__enter__()  # runs startup events (model loading)

    def post_file(self, route: str, filename: str, data: bytes, mime: str) -> Resp:
        files = {"file": (filename, data, mime)}
        if self.base_url:
            r = self._s.post(self.base_url.rstrip("/") + route, files=files, timeout=60)
        else:
            r = self._tc.post(route, files=files)
        return _wrap(r)

    def post_nothing(self, route: str) -> Resp:
        if self.base_url:
            r = self._s.post(self.base_url.rstrip("/") + route, timeout=60)
        else:
            r = self._tc.post(route)
        return _wrap(r)

    def close(self):
        if not self.base_url:
            self._tc.__exit__(None, None, None)


def _wrap(r) -> Resp:
    try:
        body = r.json()
    except Exception:
        body = None
    return Resp(r.status_code, body, r.text)


@pytest.fixture(scope="session")
def api():
    client = ApiClient(os.environ.get("GUARDRAILS_BASE_URL"))
    yield client
    client.close()


# ------------------------------------------------------------------ images --
def make_image(fmt: str, size=(320, 240), mode="RGB", color=(128, 128, 128),
               exif_orientation: int | None = None) -> bytes:
    from PIL import Image
    img = Image.new(mode, size, color if mode != "L" else 128)
    buf = io.BytesIO()
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif.tobytes()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def sample_images() -> list[Path]:
    if not IMAGES_DIR or not IMAGES_DIR.is_dir():
        return []
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------- checks --
def assert_valid_analysis(resp: Resp, where: str = "") -> dict:
    """The PRD response contract. Returns the body for further checks."""
    tag = f"[{where}] " if where else ""
    assert resp.status == 200, f"{tag}expected 200, got {resp.status}: {resp.text[:300]}"
    body = resp.body
    assert isinstance(body, dict), f"{tag}response is not a JSON object: {resp.text[:200]}"
    cat = body.get("category")
    assert cat in CATEGORIES, f"{tag}category {cat!r} not in {sorted(CATEGORIES)}"

    if cat == "cat":
        preds = body.get("predictions")
        assert isinstance(preds, list) and preds, f"{tag}cat result needs a non-empty 'predictions' list"
        assert len(preds) <= MAX_PREDICTIONS, f"{tag}{len(preds)} predictions, expected at most {MAX_PREDICTIONS}"
        allowed = CAT_BREEDS | EXTRA_BREED_LABELS
        seen = set()
        for p in preds:
            assert set(p) >= {"breed", "confidence"}, f"{tag}prediction missing keys: {p}"
            b, c = p["breed"], p["confidence"]
            assert b in allowed, (f"{tag}unknown breed label {b!r} "
                                  f"(underscores not prettified, or a dog breed leaked in?)")
            assert b not in seen, f"{tag}breed {b!r} listed twice"
            seen.add(b)
            assert isinstance(c, (int, float)) and not isinstance(c, bool), \
                f"{tag}confidence must be a number 0-1, got {c!r} (string like '87.50%'?)"
            assert 0.0 <= c <= 1.0, f"{tag}confidence {c} outside 0..1 (percent instead of fraction?)"
        confs = [p["confidence"] for p in preds]
        assert confs == sorted(confs, reverse=True), f"{tag}predictions not sorted by confidence: {confs}"
        assert sum(confs) <= 1.0 + 1e-3, f"{tag}top-{len(confs)} confidences sum to {sum(confs):.4f} > 1"
    else:
        assert body.get("message") == MESSAGES[cat], \
            f"{tag}{cat} message should be {MESSAGES[cat]!r}, got {body.get('message')!r}"
        assert not body.get("predictions"), f"{tag}non-cat result must not include breed predictions"
    return body


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
