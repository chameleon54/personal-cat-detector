#!/usr/bin/env python3
"""
Fast guardrails for the Cat Breed Detector. Run by GitHub Actions
(.github/workflows/guardrails.yml); there are no local git hooks.

Standard library only -- no TensorFlow, no model loading -- so it finishes in
about a second.

Usage:
    python tools/guardrails/check_fast.py --all      # every tracked file (what CI runs)
    python tools/guardrails/check_fast.py            # only files staged in git

Exit code 0 = pass, 1 = at least one [FAIL]. [WARN] lines never block.
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Settings you may want to tune
# ----------------------------------------------------------------------------
MAX_FILE_MB_FAIL = 95          # GitHub rejects files > 100 MB
MAX_FILE_MB_WARN = 25          # model files are ~10-30 MB; warn above this
MODEL_EXTS = {".keras", ".h5", ".tflite", ".onnx", ".pb"}

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
}
REQUIRED_MESSAGES = ["It's a dog", "It's a human", "Not a cat"]

FORBIDDEN_NAMES = {"kaggle.json", ".env", "id_rsa", "id_ed25519"}
FORBIDDEN_DIR_PARTS = {"__pycache__", ".venv", "venv", "env", "tensorflow_datasets",
                       ".ipynb_checkpoints", "node_modules"}
FORBIDDEN_SUFFIXES = {".pem", ".pyc", ".key", ".p12"}

SECRET_PATTERNS = [
    (re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'), "Kaggle API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"), "GitHub token"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "API secret key"),
    (re.compile(r"""(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*['"][^'"\s]{12,}['"]"""),
     "hard-coded secret"),
]
CONFLICT_RE = re.compile(r"^(<<<<<<< |>>>>>>> |=======\s*$)", re.M)
DEBUG_RE = re.compile(r"^\s*(breakpoint\(\)|import pdb|pdb\.set_trace\(\))", re.M)
THRESHOLD_NAME_RE = re.compile(r"THRESH|CUTOFF|MIN_.*(PROB|CONF|SCORE)|(PROB|CONF|SCORE).*MIN")

TEXT_EXTS = {".py", ".js", ".html", ".css", ".md", ".txt", ".json", ".yml",
             ".yaml", ".toml", ".cfg", ".ini", ".sh", ".env", ""}

# ----------------------------------------------------------------------------
fails: list[str] = []
warns: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)


def warn(msg: str) -> None:
    warns.append(msg)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def repo_root() -> Path:
    out = git("rev-parse", "--show-toplevel").strip()
    return Path(out) if out else Path.cwd()


ROOT = repo_root()


def find_dir(*names: str) -> Path | None:
    for n in names:
        p = ROOT / n
        if p.is_dir():
            return p
    return None


BACKEND = find_dir("Main_code", "main_code")
UI = find_dir("UI", "ui")


def file_list(all_files: bool) -> list[str]:
    if all_files:
        out = git("ls-files")
    else:
        out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [l for l in out.splitlines() if l.strip()]


def staged_bytes(path: str, all_files: bool) -> bytes | None:
    """Content as it will be committed (index), falling back to working tree."""
    if not all_files:
        r = subprocess.run(["git", "show", f":{path}"], capture_output=True)
        if r.returncode == 0:
            return r.stdout
    p = ROOT / path
    return p.read_bytes() if p.is_file() else None


def staged_size(path: str, all_files: bool) -> int:
    if not all_files:
        s = git("cat-file", "-s", f":{path}").strip()
        if s.isdigit():
            return int(s)
    p = ROOT / path
    return p.stat().st_size if p.is_file() else 0


# ----------------------------------------------------------------------------
# 1. Per-file hygiene (only the files being committed)
# ----------------------------------------------------------------------------
def check_files(files: list[str], all_files: bool) -> None:
    for f in files:
        p = Path(f)
        parts = set(p.parts)
        name = p.name.lower()
        suffix = p.suffix.lower()

        if name in FORBIDDEN_NAMES or suffix in FORBIDDEN_SUFFIXES \
                or (name.startswith(".env") and name != ".env.example"):
            fail(f"{f}: this file should never be committed (secret/build file). "
                 f"Unstage it: git rm --cached \"{f}\"")
            continue
        if parts & FORBIDDEN_DIR_PARTS:
            fail(f"{f}: lives in {sorted(parts & FORBIDDEN_DIR_PARTS)[0]}/, "
                 f"which belongs in .gitignore")
            continue

        size_mb = staged_size(f, all_files) / 1_048_576
        if size_mb > MAX_FILE_MB_FAIL:
            fail(f"{f}: {size_mb:.0f} MB is too big for GitHub (limit 100 MB). "
                 f"Use Git LFS or keep it out of git.")
            continue
        if size_mb > MAX_FILE_MB_WARN:
            warn(f"{f}: {size_mb:.0f} MB. Consider Git LFS for model files.")

        if suffix not in TEXT_EXTS:
            continue
        data = staged_bytes(f, all_files)
        if data is None:
            continue
        text = data.decode("utf-8", errors="replace")

        if CONFLICT_RE.search(text) and "<<<<<<< " in text:
            fail(f"{f}: contains merge-conflict markers (<<<<<<< / >>>>>>>)")

        for rx, label in SECRET_PATTERNS:
            m = rx.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                fail(f"{f}:{line}: looks like a {label}. Move it to an "
                     f"environment variable or an ignored file.")
                break

        if suffix == ".py":
            try:
                compile(text, f, "exec")
            except SyntaxError as e:
                fail(f"{f}:{e.lineno}: Python syntax error: {e.msg}")
                continue
            m = DEBUG_RE.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                fail(f"{f}:{line}: leftover debugger call ({m.group(1).strip()})")

        if suffix == ".js":
            check_js_syntax(f, data)


def check_js_syntax(f: str, data: bytes) -> None:
    node = shutil.which("node")
    if not node:
        return  # node is optional
    r = subprocess.run([node, "--check", "-"], input=data, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = next((l for l in err if "Error" in l), err[-1] if err else "")
        fail(f"{f}: JavaScript syntax error: {detail}")


# ----------------------------------------------------------------------------
# 2. Project contract (whole working tree; cheap static checks of the PRD/plan)
# ----------------------------------------------------------------------------
def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def check_backend() -> None:
    if BACKEND is None:
        warn("No Main_code/ (or main_code/) folder found; backend checks skipped.")
        return
    rel = BACKEND.name
    py_files = [p for p in BACKEND.glob("*.py")]
    all_py = "\n".join(read(p) for p in py_files)

    main = BACKEND / "main.py"
    if not main.is_file():
        fail(f"{rel}/main.py is missing")
    else:
        src = read(main)
        for route in ("/analyze", "/predict"):
            if not re.search(rf"""['"]{re.escape(route)}['"]""", src):
                fail(f"{rel}/main.py: route \"{route}\" not found "
                     f"(PRD needs /analyze; plan keeps /predict as an alias)")

    for msg in REQUIRED_MESSAGES:
        # accept straight or curly apostrophe, and \' escapes
        pat = re.escape(msg).replace("'", r"(?:'|\\'|’)")
        if not re.search(pat, all_py):
            fail(f"{rel}/*.py: response message \"{msg}\" not found "
                 f"(the UI and PRD expect this exact text)")

    # Threshold constants must be probabilities.
    for p in py_files:
        try:
            tree = ast.parse(read(p))
        except SyntaxError:
            continue  # reported elsewhere
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, (int, float)) \
                    and not isinstance(node.value.value, bool):
                name = node.targets[0].id
                if name.isupper() and THRESHOLD_NAME_RE.search(name):
                    v = node.value.value
                    if not (0.0 < v < 1.0):
                        fail(f"{rel}/{p.name}:{node.lineno}: {name} = {v} must be "
                             f"a probability between 0 and 1")

    names_file = BACKEND / "cat_class_names.txt"
    if names_file.is_file():
        names = [l.strip() for l in read(names_file).splitlines() if l.strip()]
        norm = [n.replace(" ", "_") for n in names]
        if len(norm) != len(set(norm)):
            fail(f"{rel}/cat_class_names.txt: duplicate breed names")
        if set(norm) != CAT_BREEDS:
            missing = sorted(CAT_BREEDS - set(norm))
            extra = sorted(set(norm) - CAT_BREEDS)
            fail(f"{rel}/cat_class_names.txt: must list exactly the 12 cat breeds. "
                 f"Missing: {missing or '-'}  Unexpected: {extra or '-'}")

    has_model = any((BACKEND / m).is_file() for m in
                    ("cat_breed_model_v2.keras", "cat_breed_model.h5"))
    if not has_model:
        warn(f"{rel}/: no cat_breed_model_v2.keras or cat_breed_model.h5 found")
    if (BACKEND / "cat_breed_model_v2.keras").is_file() and not names_file.is_file():
        fail(f"{rel}/cat_breed_model_v2.keras exists but cat_class_names.txt "
             f"is missing; breed labels would be wrong")

    req = next((p for p in (ROOT / "requirements.txt", BACKEND / "requirements.txt") if p.is_file()),
               ROOT / "requirements.txt")
    rq = req.relative_to(ROOT).as_posix()
    if not req.is_file():
        fail(f"{rq} is missing")
    else:
        lines = [l.split("#")[0].strip().lower() for l in read(req).splitlines()]
        lines = [l for l in lines if l]
        def find(pkg: str) -> str | None:
            return next((l for l in lines
                         if re.match(rf"{re.escape(pkg)}(\[.*\])?\s*([<>=!~;]|$)", l)), None)
        for pkg in ("fastapi", "uvicorn", "tensorflow", "pillow", "python-multipart"):
            if not find(pkg):
                warn(f"{rq}: '{pkg}' not listed")
        cv = find("opencv-python-headless") or find("opencv-python")
        if not cv:
            fail(f"{rq}: OpenCV missing (the human face check needs it)")
        elif "<5" not in cv.replace(" ", ""):
            fail(f"{rq}: pin OpenCV below 5 (\"{cv}\"); "
                 f"OpenCV 5 dropped the Haar face detector")
        tf = find("tensorflow")
        if tf and "==" not in tf:
            warn(f"{rq}: tensorflow is not pinned (\"{tf}\"). "
                 f"A different TF version may fail to load the .keras model.")


def check_frontend() -> None:
    if UI is None:
        warn("No UI/ folder found; frontend checks skipped.")
        return
    js_files = list(UI.rglob("*.js"))
    html_files = list(UI.rglob("*.html"))
    js = "\n".join(read(p) for p in js_files)
    html = "\n".join(read(p) for p in html_files)
    if not js_files:
        fail("UI/: no JavaScript files found")
        return

    if "/analyze" not in js:
        fail("UI/*.js: does not call the /analyze endpoint")
    if re.search(r"\bDOG_BREEDS\b", js):
        fail("UI/*.js: DOG_BREEDS is back; the backend decides the category now")
    if "AbortController" not in js:
        fail("UI/*.js: no AbortController timeout; a hung backend would spin forever")
    if not re.search(r"webp", js, re.I):
        fail("UI/*.js: WEBP is not in the allowed file types")
    if re.search(r"https?://(?!127\.0\.0\.1|localhost)[\w.-]+:\d+/(analyze|predict)", js):
        warn("UI/*.js: API_URL points at a hard-coded remote host")

    accepts = re.findall(r'accept\s*=\s*"([^"]*)"', html)
    if html_files and not accepts:
        warn("UI/*.html: file input has no accept=\"...\" filter")
    for a in accepts:
        if "image/webp" not in a and ".webp" not in a:
            fail(f"UI/*.html: accept=\"{a}\" is missing image/webp")


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="check every tracked file, not just staged ones")
    args = ap.parse_args()

    files = file_list(args.all)
    check_files(files, args.all)
    check_backend()
    check_frontend()

    scope = "all tracked files" if args.all else f"{len(files)} staged file(s)"
    print(f"guardrails: fast checks on {scope}")
    for w in warns:
        print(f"  [WARN] {w}")
    for f in fails:
        print(f"  [FAIL] {f}")
    if fails:
        print(f"\n{len(fails)} problem(s) found. Fix them, commit, and push again.")
        return 1
    print("  [ OK ] all fast checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
