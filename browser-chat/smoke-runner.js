import { createBrowserChatRuntime, inferModelFamily } from "../examples/browser-loader.mjs?v=45";

const elements = {
  status: document.querySelector("#smoke-status"),
  model: document.querySelector("#smoke-model"),
  image: document.querySelector("#smoke-image"),
  audio: document.querySelector("#smoke-audio"),
  detail: document.querySelector("#smoke-detail"),
  elapsed: document.querySelector("#smoke-elapsed"),
  output: document.querySelector("#smoke-output"),
  error: document.querySelector("#smoke-error"),
};

const startedAt = Date.now();

function setText(key, value) {
  elements[key].textContent = value ?? "";
}

function setStatus(value, className = "") {
  elements.status.textContent = value;
  elements.status.className = className;
}

function updateElapsed() {
  setText("elapsed", `${Math.round((Date.now() - startedAt) / 1000)}s`);
}

async function getLocalAuthToken() {
  if (!["localhost", "127.0.0.1"].includes(window.location.hostname)) {
    return "";
  }
  try {
    const response = await fetch("/__hf_token", { cache: "no-store" });
    if (!response.ok) {
      return "";
    }
    const payload = await response.json();
    return typeof payload?.token === "string" ? payload.token.trim() : "";
  } catch {
    return "";
  }
}

async function run() {
  const url = new URL(window.location.href);
  const modelId = url.searchParams.get("model")?.trim();
  const imageUrl = url.searchParams.get("image")?.trim();
  const audioUrl = url.searchParams.get("audio")?.trim();
  const revision = url.searchParams.get("revision")?.trim() || "";
  const prompt =
    url.searchParams.get("prompt")?.trim() ||
    (imageUrl && audioUrl
      ? "Describe the attached image and transcribe the attached audio in one short sentence."
      : imageUrl
        ? "Describe the attached image in one short sentence."
        : audioUrl
          ? "Transcribe the attached audio in one short sentence."
          : "Write one short sentence about a quiet tavern.");
  const maxNewTokens = Number.parseInt(url.searchParams.get("maxTokens") || "32", 10);

  setText("model", modelId || "(missing)");
  setText("image", imageUrl || "(none)");
  setText("audio", audioUrl || "(none)");

  if (!modelId) {
    throw new Error("Missing ?model=...");
  }
  const runtime = createBrowserChatRuntime();
  const authToken = await getLocalAuthToken();
  setStatus(imageUrl || audioUrl ? "Running multimodal smoke..." : "Running text smoke...");

  const content = [];
  if (imageUrl) {
    content.push({ type: "image", url: imageUrl });
  }
  if (audioUrl) {
    content.push({ type: "audio", url: audioUrl });
  }
  content.push({ type: "text", text: prompt });

  const output = await runtime.generate({
    modelId,
    modelFamily: inferModelFamily(modelId) || "qwen3_5",
    revision,
    messages: [
      {
        role: "user",
        content: imageUrl || audioUrl ? content : prompt,
      },
    ],
    maxNewTokens: Number.isFinite(maxNewTokens) ? maxNewTokens : 32,
    authToken,
    onProgress: (info, message) => {
      updateElapsed();
      const file = typeof info?.file === "string" ? ` ${info.file}` : "";
      setText("detail", `${message || info?.status || "Running..."}${file}`);
    },
  });

  setText("output", output || "(empty)");
  setStatus("PASS", "passed");
  updateElapsed();
}

run().catch((error) => {
  setStatus("FAIL", "failed");
  setText("error", error?.stack || error?.message || String(error));
  updateElapsed();
});
