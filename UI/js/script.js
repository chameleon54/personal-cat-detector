/* ═══════════════════════════════════════════════════════════
   Cat Breed Detector — UI/js/script.js
   FastAPI contract (Main_code/main.py):
     POST {API_URL}  multipart/form-data  field name: "file"
     → 200 { category: "cat", predictions: [{ breed, confidence }], uncertain }
                                                  confidence is 0..1, top 3
     → 200 { category: "dog" | "human" | "other", message }
     → 400 / 413 / 500 { error }
   ═══════════════════════════════════════════════════════════ */

// Same origin when served by FastAPI (http://127.0.0.1:8000/),
// otherwise (file:// or Live Server) talk to the local API directly.
const API_BASE = location.port === "8000" ? "" : "http://127.0.0.1:8000";
const API_URL = `${API_BASE}/analyze`;
const REQUEST_TIMEOUT_MS = 30000;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

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
  predBlock: document.getElementById("pred-block"),
  predList: document.getElementById("pred-list"),
  predTpl: document.getElementById("pred-row-tpl"),
};

/* ─── Non-cat results: the backend decides the category ─── */
const CATEGORY_INFO = {
  dog: {
    tag: "Dog",
    note: "This detector only identifies cat breeds. Ini anjing, bukan kucing.",
  },
  human: {
    tag: "Human",
    note: "This detector only identifies cat breeds. Ini manusia, bukan kucing.",
  },
  other: {
    tag: "Not a cat",
    note: "We couldn't find a cat in this photo. Coba foto kucing yang lebih jelas.",
  },
};

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
  return Boolean(file) && ALLOWED_TYPES.includes(file.type);
}

function setFile(file) {
  if (!isImage(file)) {
    showError("Please choose a JPG, PNG or WEBP image. · Pilih foto JPG, PNG atau WEBP.");
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
  el.predList.replaceChildren();
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

  const formData = new FormData();
  formData.append("file", selectedFile); // must match FastAPI's File(...) param

  // Give up on a hung backend instead of spinning forever.
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(API_URL, { method: "POST", body: formData, signal: controller.signal });

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
    if (err.name === "AbortError") {
      showError("The server took too long to answer. Please try again. · Server terlalu lama merespons.");
    } else if (err instanceof TypeError) {
      showError("Can't reach the API. Start it with: python -m uvicorn main:app --reload (from Main_code).");
    } else {
      showError(err.message);
    }
  } finally {
    window.clearTimeout(timer);
    el.predictBtn.disabled = false;
    el.clearBtn.disabled = false;
  }
}

function renderResult(data) {
  if (data.category === "cat" && Array.isArray(data.predictions) && data.predictions.length) {
    renderCat(data);
  } else {
    renderNonCat(data);
  }
  showState("card");
  setStatus("Done", "is-done");
}

function renderCat(data) {
  const top = data.predictions[0];
  el.breedTag.textContent = "Cat breed";
  el.breedTag.className = "tag is-cat";
  el.breedName.textContent = data.uncertain ? "Mixed / uncertain breed" : top.breed;
  el.breedNote.textContent = data.uncertain
    ? `Closest match is ${top.breed}, but the model isn't sure. Bisa jadi kucing campuran.`
    : "Identified from your photo. Hasil ini perkiraan, bukan diagnosis.";

  const rows = data.predictions.map(({ breed, confidence }) => {
    const row = el.predTpl.content.firstElementChild.cloneNode(true);
    const pct = toPercent(confidence);
    row.querySelector(".pred-breed").textContent = breed;
    row.querySelector(".pred-value").textContent = `${pct.toFixed(1)}%`;
    row.dataset.pct = pct;
    return row;
  });
  el.predList.replaceChildren(...rows);
  el.predBlock.classList.remove("hidden");

  // Animate the bars from 0 once the rows are on screen.
  requestAnimationFrame(() => {
    rows.forEach((row) => {
      row.querySelector(".meter-fill").style.width = `${row.dataset.pct}%`;
    });
  });
}

function renderNonCat(data) {
  const info = CATEGORY_INFO[data.category] ?? CATEGORY_INFO.other;
  el.breedTag.textContent = info.tag;
  el.breedTag.className = `tag is-${data.category in CATEGORY_INFO ? data.category : "other"}`;
  el.breedName.textContent = data.message ?? "Not a cat";
  el.breedNote.textContent = info.note;
  el.predList.replaceChildren();
  el.predBlock.classList.add("hidden");
}

// Backend sends 0..1 — convert to a clamped percentage.
function toPercent(confidence) {
  const n = Number(confidence) * 100;
  return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : 0;
}
