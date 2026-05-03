import { createBrowserChatRuntime, inferModelFamily } from "../examples/browser-loader.mjs?v=39";

const elements = {
  status: document.querySelector("#smoke-status"),
  model: document.querySelector("#smoke-model"),
  image: document.querySelector("#smoke-image"),
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
  const prompt =
    url.searchParams.get("prompt")?.trim() ||
    "Describe the attached image in one short sentence.";
  const maxNewTokens = Number.parseInt(url.searchParams.get("maxTokens") || "32", 10);

  setText("model", modelId || "(missing)");
  setText("image", imageUrl || "(none)");

  if (!modelId) {
    throw new Error("Missing ?model=...");
  }
  if (!imageUrl) {
    throw new Error("Missing ?image=...");
  }

  const runtime = createBrowserChatRuntime();
  const authToken = await getLocalAuthToken();
  setStatus("Running multimodal smoke...");

  const output = await runtime.generate({
    modelId,
    modelFamily: inferModelFamily(modelId) || "qwen3_5",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image", url: imageUrl },
        ],
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
