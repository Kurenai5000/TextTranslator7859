const fileInput = document.getElementById("fileInput");
const apiInput = document.getElementById("apiInput");
const translateBtn = document.getElementById("translateBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusEl = document.getElementById("status");
const progressFill = document.getElementById("progressFill");
const previewEl = document.getElementById("preview");

const CONFIG = {
  API_URL: "https://api.deepseek.com/chat/completions",
  MODEL: "deepseek-chat",
  MAX_TOKENS_PER_CHUNK: 2000,
  CHARS_PER_TOKEN: 2.5,
  MAX_RETRIES: 4,
  RETRY_BASE_DELAY_MS: 1000,
  RATE_LIMIT_DELAY_MS: 300,
  SAFETY_MARGIN: 0.85,
  SYSTEM_PROMPT: "You are a translator. Translate the user's text faithfully, preserving meaning, tone, and structure. Output only the translation, with no commentary.",
};

// State
let file = null;
let source = "";
let position = 0;
let output = "";
let isTranslating = false;

// Button state
function updateButtonState() {
  const hasFile = !!file;
  const hasKey = apiInput.value.trim().length > 0;
  translateBtn.disabled = !(hasFile && hasKey) || isTranslating;
  downloadBtn.disabled = output.length === 0 || isTranslating;
}

fileInput.addEventListener("change", () => {
  file = fileInput.files[0] || null;
  updateButtonState();
});

apiInput.addEventListener("input", updateButtonState);

updateButtonState();

// UI helpers
function setStatus(msg) {
  statusEl.textContent = msg;
}

function updateProgress() {
  if (source.length === 0) {
    progressFill.style.width = "0%";
    return;
  }
  const pct = Math.min(100, (position / source.length) * 100);
  progressFill.style.width = pct.toFixed(1) + "%";
}

function updatePreview() {
  const tail = output.slice(-500);
  previewEl.textContent = tail || "(no output yet)";
  previewEl.scrollTop = previewEl.scrollHeight;
}

// Chunking helpers
function findLastParagraphBreak(text) {
  let idx = text.lastIndexOf("\n\n");
  if (idx !== -1) return idx + 2;
  idx = text.lastIndexOf("。\n");
  if (idx !== -1) return idx + 2;
  idx = text.lastIndexOf("\n");
  if (idx !== -1) return idx + 1;
  return -1;
}

function findLastSentenceBreak(text) {
  const markers = [". ", "! ", "? ", "。", "！", "？", "；", "; "];
  let best = -1;
  for (const m of markers) {
    const idx = text.lastIndexOf(m);
    if (idx > best) best = idx + m.length;
  }
  return best;
}

function getNextChunk(source, position, maxTokens) {
  const maxChars = Math.floor(maxTokens * CONFIG.CHARS_PER_TOKEN * CONFIG.SAFETY_MARGIN);
  const remaining = source.slice(position);

  if (remaining.length <= maxChars) {
    return { text: remaining, end: source.length };
  }

  const window = remaining.slice(0, maxChars);

  const paraBreak = findLastParagraphBreak(window);
  if (paraBreak > 0) {
    return { text: window.slice(0, paraBreak), end: position + paraBreak };
  }

  const sentBreak = findLastSentenceBreak(window);
  if (sentBreak > 0) {
    return { text: window.slice(0, sentBreak), end: position + sentBreak };
  }

  return { text: window, end: position + maxChars };
}

function estimateTokens(text) {
  return Math.ceil(text.length / CONFIG.CHARS_PER_TOKEN);
}

// Async helpers
async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function translateChunk(text, apiKey) {
  const response = await fetch(CONFIG.API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: CONFIG.MODEL,
      messages: [
        { role: "system", content: CONFIG.SYSTEM_PROMPT },
        { role: "user", content: text },
      ],
      temperature: 0.3,
    }),
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => "");
    const err = new Error(`API ${response.status}: ${response.statusText} ${errText}`);
    err.status = response.status;
    throw err;
  }

  const data = await response.json();
  const translated = data?.choices?.[0]?.message?.content;
  if (typeof translated !== "string") {
    throw new Error("Malformed API response");
  }
  return translated;
}

async function translateChunkWithRetry(text, apiKey) {
  let lastErr = null;
  for (let attempt = 1; attempt <= CONFIG.MAX_RETRIES; attempt++) {
    try {
      return await translateChunk(text, apiKey);
    } catch (err) {
      lastErr = err;
      if (err.status === 401 || err.status === 403) {
        throw err;
      }
      const delay = CONFIG.RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1);
      setStatus(`Chunk failed (attempt ${attempt}/${CONFIG.MAX_RETRIES}). Retrying in ${delay}ms...`);
      await sleep(delay);
    }
  }
  throw lastErr;
}

// Checkpointing
const CHECKPOINT_KEY = "translator_checkpoint_v1";

function saveCheckpoint() {
  try {
    localStorage.setItem(
      CHECKPOINT_KEY,
      JSON.stringify({
        position,
        output,
        sourceLength: source.length,
        timestamp: Date.now(),
      })
    );
  } catch (err) {
    console.warn("Could not save checkpoint:", err);
  }
}

function loadCheckpoint() {
  try {
    const raw = localStorage.getItem(CHECKPOINT_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function clearCheckpoint() {
  try {
    localStorage.removeItem(CHECKPOINT_KEY);
  } catch {}
}

// Main loop
async function translateAll() {
  const apiKey = apiInput.value.trim();
  if (!file || !apiKey) return;

  isTranslating = true;
  position = 0;
  output = "";
  updateButtonState();
  updateProgress();
  updatePreview();

  try {
    setStatus("Reading file...");
    source = await file.text();

    if (source.length === 0) {
      setStatus("File is empty.");
      return;
    }

    setStatus(`Translating ${source.length} characters...`);

    let chunkCount = 0;
    while (position < source.length) {
      const chunk = getNextChunk(source, position, CONFIG.MAX_TOKENS_PER_CHUNK);
      chunkCount++;

      setStatus(
        `Chunk ${chunkCount} — chars ${position}–${chunk.end} of ${source.length} (est. ${estimateTokens(chunk.text)} tokens)`
      );

      let translated = null;
      try {
        translated = await translateChunkWithRetry(chunk.text, apiKey);
      } catch (err) {
        console.error("Chunk permanently failed:", err);
        setStatus(`Chunk failed permanently at position ${position}. Halting. Progress saved.`);
        saveCheckpoint();
        return;
      }

      output += translated;
      if (!output.endsWith("\n")) output += "\n";

      position = chunk.end;

      updateProgress();
      updatePreview();
      saveCheckpoint();

      await sleep(CONFIG.RATE_LIMIT_DELAY_MS);
    }

    setStatus(`Done. Translated ${chunkCount} chunks, ${output.length} characters.`);
    clearCheckpoint();
    downloadBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("Error: " + err.message);
    saveCheckpoint();
  } finally {
    isTranslating = false;
    updateButtonState();
  }
}

// Event listeners
translateBtn.addEventListener("click", translateAll);

downloadBtn.addEventListener("click", () => {
  const blob = new Blob([output], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = (file?.name?.replace(/\.[^.]+$/, "") || "translation") + "_translated.txt";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});
