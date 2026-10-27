"""
API guardrails: response contract, input validation, robustness, latency.
These use generated images, so they work with any model version.
"""
import time

import pytest

from conftest import (MAX_LATENCY_S, MIME, assert_valid_analysis, make_image,
                      sample_images)

ROUTES = ["/analyze", "/predict"]


# ---------------------------------------------------------------- formats --
VALID_UPLOADS = {
    "jpeg":            lambda: ("photo.jpg",  make_image("JPEG"),  "image/jpeg"),
    "png-rgb":         lambda: ("photo.png",  make_image("PNG"),   "image/png"),
    "png-rgba":        lambda: ("alpha.png",  make_image("PNG", mode="RGBA", color=(90, 90, 90, 128)), "image/png"),
    "png-grayscale":   lambda: ("gray.png",   make_image("PNG", mode="L"), "image/png"),
    "webp":            lambda: ("photo.webp", make_image("WEBP"),  "image/webp"),
    "tiny-1x1":        lambda: ("tiny.png",   make_image("PNG", size=(1, 1)), "image/png"),
    "huge-4000x3000":  lambda: ("huge.jpg",   make_image("JPEG", size=(4000, 3000)), "image/jpeg"),
    "portrait":        lambda: ("tall.jpg",   make_image("JPEG", size=(240, 900)), "image/jpeg"),
    "exif-rotated":    lambda: ("phone.jpg",  make_image("JPEG", exif_orientation=6), "image/jpeg"),
    "uppercase-ext":   lambda: ("PHOTO.JPG",  make_image("JPEG"),  "image/jpeg"),
}


@pytest.mark.parametrize("kind", list(VALID_UPLOADS))
def test_valid_image_formats_return_contract(api, kind):
    name, data, mime = VALID_UPLOADS[kind]()
    assert_valid_analysis(api.post_file("/analyze", name, data, mime), kind)


def test_blank_image_is_not_a_cat(api):
    """A plain grey picture must not be sold to the user as a cat breed."""
    body = assert_valid_analysis(
        api.post_file("/analyze", "blank.jpg", make_image("JPEG", color=(200, 200, 200)), "image/jpeg"))
    assert body["category"] in {"other", "human"}, \
        f"blank image classified as {body['category']!r}; gatekeeper threshold too low?"


# ------------------------------------------------------- invalid uploads --
INVALID_UPLOADS = {
    "text-file":       ("notes.txt", b"hello, not an image", "text/plain"),
    "empty-file":      ("empty.jpg", b"", "image/jpeg"),
    "corrupt-jpeg":    ("broken.jpg", b"\xff\xd8\xff\xe0" + b"\x00garbage" * 50, "image/jpeg"),
    "text-as-jpg":     ("fake.jpg", b"this is plain text pretending to be a jpg", "image/jpeg"),
    "pdf":             ("doc.pdf", b"%PDF-1.4\n%fake\n", "application/pdf"),
    "gif":             ("anim.gif", None, "image/gif"),   # filled below: a real GIF, not allowed by PRD
}


@pytest.mark.parametrize("kind", list(INVALID_UPLOADS))
def test_invalid_upload_is_a_clean_400(api, kind):
    name, data, mime = INVALID_UPLOADS[kind]
    if data is None:
        data = make_image("GIF")
    r = api.post_file("/analyze", name, data, mime)
    assert r.status != 500, f"{kind}: server crashed (500) instead of rejecting: {r.text[:300]}"
    assert r.status == 400, f"{kind}: expected 400, got {r.status}: {r.text[:300]}"
    assert isinstance(r.body, dict) and r.body.get("error"), \
        f"{kind}: 400 response should be {{\"error\": \"...\"}}, got {r.text[:200]}"


def test_missing_file_field_is_rejected_not_crash(api):
    r = api.post_nothing("/analyze")
    assert 400 <= r.status < 500, f"no file sent: expected 4xx, got {r.status}"


# ----------------------------------------------------- routes & stability --
def _a_real_or_fake_image():
    imgs = sample_images()
    if imgs:
        p = imgs[0]
        return p.name, p.read_bytes(), MIME[p.suffix.lower()]
    return "photo.jpg", make_image("JPEG"), "image/jpeg"


@pytest.mark.parametrize("route", ROUTES)
def test_route_exists(api, route):
    name, data, mime = _a_real_or_fake_image()
    assert_valid_analysis(api.post_file(route, name, data, mime), route)


def test_predict_alias_matches_analyze(api):
    name, data, mime = _a_real_or_fake_image()
    a = api.post_file("/analyze", name, data, mime).body
    b = api.post_file("/predict", name, data, mime).body
    assert a == b, f"/predict and /analyze disagree:\n  {a}\n  {b}"


def test_same_image_gives_same_answer(api):
    """Catches augmentation layers accidentally active at inference time."""
    name, data, mime = _a_real_or_fake_image()
    first = api.post_file("/analyze", name, data, mime).body
    for _ in range(2):
        again = api.post_file("/analyze", name, data, mime).body
        assert again == first, f"non-deterministic result:\n  {first}\n  {again}"


def test_latency_after_warmup(api):
    name, data, mime = _a_real_or_fake_image()
    api.post_file("/analyze", name, data, mime)          # warm-up (model load, graph build)
    t0 = time.perf_counter()
    api.post_file("/analyze", name, data, mime)
    dt = time.perf_counter() - t0
    assert dt < MAX_LATENCY_S, (f"one request took {dt:.2f}s (limit {MAX_LATENCY_S}s; "
                                f"raise with GUARDRAILS_MAX_LATENCY_S)")
