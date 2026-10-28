# Guardrails (GitHub only)

Automatic checks that run **on GitHub** for every push and pull request. They stop broken code, leaked secrets
and model regressions from reaching `main`. Nothing runs on your own PC: there are no git hooks, so
`git commit` and `git push` stay instant.

Workflow file: [.github/workflows/guardrails.yml](.github/workflows/guardrails.yml)

| Job | What it does | Time |
|---|---|---|
| **Fast checks** | `tools/guardrails/check_fast.py --all` (no TensorFlow) | ~20 s |
| **API & model tests** | installs `requirements.txt` + `requirements-dev.txt`, runs `pytest tests/` | ~5-8 min first run, faster with cache |

The tests job only starts if the fast checks pass.

## What the fast checks look for
- Secrets: `kaggle.json`, `.env`, API keys/tokens, private keys
- Junk in git: `__pycache__/`, `.pyc`, venvs, `tensorflow_datasets/`
- Files over 95 MB (GitHub's limit is 100 MB); warning above 25 MB
- Python and JavaScript syntax errors, leftover `breakpoint()` / `pdb`, merge-conflict markers
- The project contract from the PRD:
  - `main.py` has both `/analyze` and `/predict`
  - the exact messages "It's a dog", "It's a human", "Not a cat" exist in `Main_code/*.py`
  - threshold constants (names containing `THRESH` / `CUTOFF`) are between 0 and 1
  - `cat_class_names.txt` lists exactly the 12 cat breeds, and ships with `cat_breed_model_v2.keras`
  - `requirements.txt` pins OpenCV below 5
  - the UI calls `/analyze`, uses `AbortController`, allows WEBP, and has no `DOG_BREEDS`

## What the tests check
`tests/test_api.py` (generated images, works with any model):
- every valid upload (JPG, PNG, RGBA, grayscale, WEBP, 1x1, 4000x3000, EXIF-rotated) returns the PRD JSON shape
- cat results: up to 3 predictions, known breed names, confidences 0-1, sorted
- dog/human/other results: the exact message and no breed list
- bad uploads (text, empty, corrupt JPEG, PDF, GIF) get a clean `400`, never a `500`
- `/predict` gives the same answer as `/analyze`; repeated calls agree; a blank grey image is not called a cat
- one warm request finishes in time (`GUARDRAILS_MAX_LATENCY_S`, 10 s on GitHub)

`tests/test_golden.py` (your real photos in `Main_code/Images/`):
- each photo's category must not change from `tests/golden.json`
- a cat's recorded breed must stay in the top 3 (a top-1 change only warns)
- file names starting with `cat`, `dog`, `human`, `person`, `selfie` must land in that category

## After you retrain on purpose
The golden file records what the model says today. After a new model you are happy with:
```
python -m pip install -r requirements-dev.txt
UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py     # PowerShell: $env:UPDATE_GOLDEN=1
```
Check `tests/golden.json`, commit it, push.

If you add a "mixed / uncertain" breed label to the API, list it in `EXTRA_BREED_LABELS` in `tests/conftest.py`.

## Make it block merges (one-time, on github.com)
1. Push the workflow and let it run once. The **Actions** tab should show **Fast checks** and **API & model tests** green.
   (GitHub only offers a check in the rule after it has run once.)
2. Repo **Settings -> Rules -> Rulesets -> New ruleset -> New branch ruleset**.
3. Name it `main guardrails`, set **Enforcement status** to **Active**.
4. **Target branches -> Add target -> Include default branch** (that is `main`).
5. Turn on **Require a pull request before merging**.
6. Turn on **Require status checks to pass**, then **Add checks** and pick **Fast checks** and **API & model tests**.
7. Turn on **Block force pushes**. Click **Create**.

From then on `main` only changes through a pull request whose checks passed (for example `dev_Web` -> `main`).
Leave the **Bypass list** empty so nobody can skip it, or add yourself if you want an emergency override.

Rulesets and branch protection on a **private** repo need a paid GitHub plan. On a free private repo the checks still
run and show a red X, but cannot block a merge. Public repos get it free.

## Run the same checks by hand (optional)
```
python tools/guardrails/check_fast.py --all
python -m pytest tests -v
GUARDRAILS_BASE_URL=http://127.0.0.1:8000 python -m pytest tests   # against a running server
```
