/* ═══════════════════════════════════════════════════════════
   Cat Breed Detector — UI/js/script.js
   FastAPI contract (Main_code/main.py):
     POST {API_URL}  multipart/form-data  field name: "file"
     → 200 { filename, label, confidence }        confidence like "87.50%"
     → 500 { error }
   ═══════════════════════════════════════════════════════════ */

const API_URL = "http://127.0.0.1:8000/predict";

/* ─── Element refs ─── */
const el = {
  home: document.getElementById("view-home"),
  detect: document.getElementById("view-detect"),
  startBtn: document.getElementById("start-btn"),
  backBtn: document.getElementById("back-btn"),
  againBtn: document.getElementById("again-btn"),

  dropzone: document.getElementById("drop-zone"),
  input: document.getElementById("imageInput"),
  dropEmpty: document.getElementById("drop-empty"),
  dropPreview: document.getElementById("drop-preview"),
  previewImg: document.getElementById("preview-img"),
  removeBtn: document.getElementById("remove-btn"),

  fileMeta: document.getElementById("file-meta"),
  clearBtn: document.getElementById("clear-btn"),
  predictBtn: document.getElementById("predict-btn"),

  panel: document.getElementById("result-panel"),
  status: document.getElementById("result-status"),
  placeholder: document.getElementById("result-placeholder"),
  loading: document.getElementById("result-loading"),
  error: document.getElementById("result-error"),
  errorText: document.getElementById("error-text"),
  card: document.getElementById("result-card"),
  breedTag: document.getElementById("breed-tag"),
  breedName: document.getElementById("breed-name"),
  breedNote: document.getElementById("breed-note"),
  confValue: document.getElementById("conf-value"),
  confFill: document.getElementById("conf-fill"),
};

/* ─── The model is NOT cats-only: Oxford-IIIT Pet includes dog breeds ─── */
const DOG_BREEDS = new Set([
  "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle", "boxer",
  "chihuahua", "english_cocker_spaniel", "english_setter", "german_shorthaired",
  "great_pyrenees", "havanese", "japanese_chin", "keeshond", "leonberger",
  "miniature_pinscher", "newfoundland", "pomeranian", "pug", "saint_bernard",
  "samoyed", "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
  "wheaten_terrier", "yorkshire_terrier",
]);

let selectedFile = null;

/* ═══════════════ View switching ═══════════════ */
function showView(target) {
  const from = target === "detect" ? el.home : el.detect;
  const to = target === "detect" ? el.detect : el.home;

  if (!to.hidden) return; // already there

  from.classList.add("is-leaving");
  from.setAttribute("aria-hidden", "true");

  window.setTimeout(() => {
    from.hidden = true;
    from.classList.remove("is-leaving");
    from.setAttribute("aria-hidden", "false");

    to.hidden = false;
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });

    // Restart the entrance animation for the freshly shown view.
    to.classList.remove("view");
    void to.offsetWidth;
    to.classList.add("view");
  }, 250);
}

el.startBtn.addEventListener("click", () => showView("detect"));
el.backBtn.addEventListener("click", () => showView("home"));
el.againBtn.addEventListener("click", () => {
  resetSelection();
  el.dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
});

/* ═══════════════ File selection ═══════════════ */
function isImage(file) {
  return Boolean(file) && file.type.startsWith("image/");
}

function setFile(file) {
  if (!isImage(file)) {
    showError("That file isn't an image. Please choose a JPG or PNG.");
    return;
  }
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    el.previewImg.src = e.target.result;
    el.dropEmpty.classList.add("hidden");
    el.dropPreview.classList.remove("hidden");
    el.dropzone.classList.add("has-image");
    el.dropzone.setAttribute("aria-label", "Selected photo. Click remove to choose another.");
  };
  reader.readAsDataURL(file);

  el.fileMeta.textContent = `${file.name} · ${formatSize(file.size)}`;
  el.clearBtn.disabled = false;
  el.predictBtn.disabled = false;
  showState("placeholder");
  setStatus("Ready");
}

function resetSelection() {
  selectedFile = null;
  el.input.value = "";
  el.previewImg.removeAttribute("src");
  el.dropPreview.classList.add("hidden");
  el.dropEmpty.classList.remove("hidden");
  el.dropzone.classList.remove("has-image");
  el.dropzone.setAttribute("aria-label", "Upload a cat photo");

  el.fileMeta.textContent = "No file selected · Belum ada foto";
  el.clearBtn.disabled = true;
  el.predictBtn.disabled = true;
  el.confFill.style.width = "0%";
  showState("placeholder");
  setStatus("Idle");
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/* Native input (label-less zone: we forward clicks ourselves) */
el.input.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
});

el.dropzone.addEventListener("click", () => {
  if (!el.dropzone.classList.contains("has-image")) el.input.click();
});
el.dropzone.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && !el.dropzone.classList.contains("has-image")) {
    e.preventDefault();
    el.input.click();
  }
});

el.removeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  resetSelection();
});
el.clearBtn.addEventListener("click", resetSelection);

/* ═══════════════ Drag & drop ═══════════════ */
["dragenter", "dragover"].forEach((evt) =>
  el.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    el.dropzone.classList.add("is-dragging");
  })
);
["dragleave", "drop"].forEach((evt) =>
  el.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    el.dropzone.classList.remove("is-dragging");
  })
);
el.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) setFile(file);
});

// Don't let the browser open an image dropped outside the zone.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

/* ═══════════════ Result panel state machine ═══════════════ */
const STATES = ["placeholder", "loading", "error", "card"];
function showState(name) {
  STATES.forEach((key) => el[key].classList.toggle("hidden", key !== name));
}

function setStatus(text, modifier = "") {
  el.status.textContent = text;
  el.status.className = "status-dot" + (modifier ? ` ${modifier}` : "");
}

function showError(message) {
  el.errorText.textContent = message;
  showState("error");
  setStatus("Error", "is-error");
}

/* ═══════════════ Prediction ═══════════════ */
el.predictBtn.addEventListener("click", predict);

async function predict() {
  if (!selectedFile) {
    showError("Please choose an image first. · Pilih foto dulu.");
    return;
  }

  el.predictBtn.disabled = true;
  el.clearBtn.disabled = true;
  showState("loading");
  setStatus("Working", "is-working");
  el.confFill.style.width = "0%";

  const formData = new FormData();
  formData.append("file", selectedFile); // must match FastAPI's File(...) param

  try {
    const res = await fetch(API_URL, { method: "POST", body: formData });

    // Backend returns JSON for both success and error paths.
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error(`Server returned ${res.status}. Is the API running?`);
    }

    if (!res.ok || data.error) {
      throw new Error(data.error || `Server returned ${res.status}.`);
    }

    renderResult(data);
  } catch (err) {
    const isNetwork = err instanceof TypeError;
    showError(
      isNetwork
        ? "Can't reach the API. Start it with: python -m uvicorn main:app --reload (from Main_code)."
        : err.message
    );
  } finally {
    el.predictBtn.disabled = false;
    el.clearBtn.disabled = false;
  }
}

function renderResult(data) {
  const raw = String(data.label ?? "Unknown");
  const isDog = DOG_BREEDS.has(raw.toLowerCase());
  const pretty = prettify(raw);

  // Title-case the cat breeds; dog breeds read better lowercase in a sentence.
  el.breedName.textContent = isDog ? pretty : titleCase(pretty);

  el.breedTag.textContent = isDog ? "Dog breed — model also detects dogs" : "Cat breed";
  el.breedTag.className = "tag " + (isDog ? "is-dog" : "is-cat");

  el.breedNote.textContent = isDog
    ? "Trained on Oxford-IIIT Pet, so it knows dog breeds too. Ras ini termasuk anjing."
    : "Identified from your photo. Hasil ini perkiraan, bukan diagnosis.";

  // Backend sends "87.50%" (a string) — normalize before doing math.
  const pct = toPercent(data.confidence);
  el.confValue.textContent = `${pct.toFixed(2)}%`;

  showState("card");
  setStatus("Done", "is-done");

  requestAnimationFrame(() => {
    el.confFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  });
}

function toPercent(value) {
  const n = parseFloat(String(value).replace("%", "").trim());
  return Number.isFinite(n) ? n : 0;
}

function prettify(label) {
  return String(label).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}
function titleCase(text) {
  return text.replace(/\b\w/g, (c) => c.toUpperCase());
}
