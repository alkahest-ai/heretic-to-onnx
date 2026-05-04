import {
  DEFAULT_MODEL_ID,
  DEFAULT_MODEL_PRESETS,
  findModelPreset,
  formatPresetSummary,
} from "../examples/browser-loader.mjs?v=40";
import { formatRuntimeError } from "./runtime-errors.mjs";
import { createBrowserChatRuntimeClient } from "./runtime-client.js?v=35";

const elements = {
  presetModel: document.querySelector("#preset-model"),
  modelId: document.querySelector("#model-id"),
  presetNote: document.querySelector("#preset-note"),
  hfToken: document.querySelector("#hf-token"),
  imageInput: document.querySelector("#image-input"),
  imageStatus: document.querySelector("#image-status"),
  audioInput: document.querySelector("#audio-input"),
  audioStatus: document.querySelector("#audio-status"),
  videoInput: document.querySelector("#video-input"),
  videoStatus: document.querySelector("#video-status"),
  systemPrompt: document.querySelector("#system-prompt"),
  maxTokens: document.querySelector("#max-tokens"),
  temperature: document.querySelector("#temperature"),
  loadModel: document.querySelector("#load-model"),
  clearChat: document.querySelector("#clear-chat"),
  clearCache: document.querySelector("#clear-cache"),
  runtimeStatus: document.querySelector("#runtime-status"),
  runtimePercent: document.querySelector("#runtime-percent"),
  runtimeAge: document.querySelector("#runtime-age"),
  runtimeProgressBar: document.querySelector("#runtime-progress-bar"),
  runtimeDetail: document.querySelector("#runtime-detail"),
  runtimeLog: document.querySelector("#runtime-log"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chat-form"),
  promptInput: document.querySelector("#prompt-input"),
  sendMessage: document.querySelector("#send-message"),
};

const state = {
  runtime: createBrowserChatRuntimeClient(),
  messages: [],
  loading: false,
  generating: false,
  runtimeReady: false,
  loadedModelId: null,
  loadedTextOnly: null,
  composerMedia: {
    image: null,
    audio: null,
    video: null,
  },
};

const statusState = {
  timer: null,
  pending: null,
  lastRendered: "",
};

let localAuthTokenPromise = null;

const runtimeState = {
  percent: null,
  detail: "Waiting for model activity.",
  lastUpdateAt: null,
  logEntries: [],
  lastLogKey: "",
};

const pendingDotsState = {
  frame: 0,
  timer: null,
};

const HF_TOKEN_STORAGE_KEY = "heretic.browserChat.hfToken";

function initialModelId() {
  const url = new URL(window.location.href);
  return url.searchParams.get("model") || DEFAULT_MODEL_ID;
}

function getAuthToken() {
  return elements.hfToken.value.trim();
}

function initializeAuthToken() {
  const stored = window.localStorage.getItem(HF_TOKEN_STORAGE_KEY);
  if (stored) {
    elements.hfToken.value = stored;
  }
}

async function discoverLocalAuthToken() {
  if (getAuthToken()) {
    return;
  }
  if (!["localhost", "127.0.0.1"].includes(window.location.hostname)) {
    return;
  }

  try {
    const response = await fetch("/__hf_token", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    const token = typeof payload?.token === "string" ? payload.token.trim() : "";
    if (!token) {
      return;
    }
    elements.hfToken.value = token;
    window.localStorage.setItem(HF_TOKEN_STORAGE_KEY, token);
    pushRuntimeLog("Loaded local HF token for private repo access.", "local-hf-token");
  } catch {
    // Local token discovery is best-effort; manual paste still works.
  }
}

function initializeLocalAuthToken() {
  localAuthTokenPromise ||= discoverLocalAuthToken();
  return localAuthTokenPromise;
}

function clampPercent(value) {
  return Math.max(0, Math.min(100, value));
}

function formatRuntimePercent(percent) {
  return percent == null ? "--%" : `${Math.round(clampPercent(percent))}%`;
}

function formatByteCount(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return null;
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 100 || unitIndex === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function formatRuntimeAge(timestamp) {
  if (!timestamp) {
    return "No transfers yet.";
  }
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 2) {
    return "Updated just now.";
  }
  if (seconds < 60) {
    return `Updated ${seconds}s ago.`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder > 0 ? `Updated ${minutes}m ${remainder}s ago.` : `Updated ${minutes}m ago.`;
}

function formatLogTime(date) {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function renderRuntimeLog() {
  elements.runtimeLog.innerHTML = "";

  if (runtimeState.logEntries.length === 0) {
    const line = document.createElement("p");
    line.className = "runtime-log-entry";
    line.textContent = "No runtime events yet.";
    elements.runtimeLog.append(line);
    return;
  }

  for (const entry of runtimeState.logEntries) {
    const line = document.createElement("p");
    line.className = "runtime-log-entry";

    const stamp = document.createElement("strong");
    stamp.textContent = formatLogTime(entry.time);
    line.append(stamp);
    line.append(` ${entry.text}`);
    elements.runtimeLog.append(line);
  }
}

function renderRuntimeAge() {
  elements.runtimeAge.textContent = formatRuntimeAge(runtimeState.lastUpdateAt);
}

function renderRuntimePanel() {
  const statusText = (statusState.pending ?? statusState.lastRendered) || "Waiting for model activity.";
  elements.runtimeStatus.textContent = statusText;
  elements.runtimePercent.textContent = formatRuntimePercent(runtimeState.percent);
  renderRuntimeAge();
  elements.runtimeProgressBar.style.width =
    runtimeState.percent == null ? "0%" : `${clampPercent(runtimeState.percent)}%`;
  elements.runtimeDetail.textContent = runtimeState.detail;
}

function pushRuntimeLog(text, key = text) {
  if (!text || key === runtimeState.lastLogKey) {
    return;
  }
  runtimeState.lastLogKey = key;
  runtimeState.logEntries = [{ time: new Date(), text }, ...runtimeState.logEntries].slice(0, 12);
  renderRuntimeLog();
}

function beginRuntimeActivity(statusMessage, { detail = statusMessage, resetProgress = false } = {}) {
  state.runtimeReady = false;
  if (resetProgress) {
    runtimeState.percent = 0;
  }
  runtimeState.detail = detail;
  runtimeState.lastUpdateAt = Date.now();
  pushRuntimeLog(statusMessage);
  setStatus(statusMessage, { immediate: true });
  renderRuntimePanel();
}

function finishRuntimeActivity(statusMessage, { detail = statusMessage, percent = 100 } = {}) {
  state.runtimeReady = true;
  runtimeState.percent = percent;
  runtimeState.detail = detail;
  runtimeState.lastUpdateAt = Date.now();
  pushRuntimeLog(statusMessage);
  setStatus(statusMessage, { immediate: true });
  renderRuntimePanel();
}

function failRuntimeActivity(statusMessage) {
  state.runtimeReady = false;
  runtimeState.detail = statusMessage;
  runtimeState.lastUpdateAt = Date.now();
  pushRuntimeLog(statusMessage);
  setStatus(statusMessage, { immediate: true });
  renderRuntimePanel();
}

function extractProgressPercent(info) {
  if (typeof info?.progress === "number") {
    return clampPercent(info.progress);
  }
  if (typeof info?.loaded === "number" && typeof info?.total === "number" && info.total > 0) {
    return clampPercent((info.loaded / info.total) * 100);
  }
  return null;
}

function buildRuntimeDetail(info, message) {
  const percent = extractProgressPercent(info);
  const file = typeof info?.file === "string" ? info.file : "";
  const loaded = formatByteCount(info?.loaded);
  const total = formatByteCount(info?.total);
  const bytes = loaded && total ? `${loaded} / ${total}` : loaded ? `${loaded} transferred` : "";
  const status = typeof info?.status === "string" ? info.status.replaceAll("_", " ") : "";

  if (info?.status === "done") {
    return {
      percent: 100,
      detail: message || "Download complete. Initializing WebGPU sessions...",
      logText: message || "Download complete. Initializing WebGPU sessions...",
      logKey: "done",
    };
  }

  const detailParts = [];
  if (file) {
    detailParts.push(file);
  }
  if (percent != null) {
    detailParts.push(`${Math.round(percent)}%`);
  }
  if (bytes) {
    detailParts.push(bytes);
  }
  if (detailParts.length === 0 && status) {
    detailParts.push(status);
  }
  if (detailParts.length === 0) {
    detailParts.push(message || "Loading model...");
  }

  const bucket = percent == null ? "na" : String(Math.floor(percent / 10));
  const logText = file
    ? `${status || "progress"}: ${file}${percent != null ? ` (${Math.round(percent)}%)` : ""}${
        bytes ? `, ${bytes}` : ""
      }`
    : message || status || "Loading model...";
  const logKey = `${status}|${file}|${bucket}`;

  return {
    percent,
    detail: detailParts.join(" | "),
    logText,
    logKey,
  };
}

function handleRuntimeProgress(info, message) {
  const nextStatus = message || "Loading model...";
  const next = buildRuntimeDetail(info, nextStatus);
  if (info?.status !== "sessions_warm") {
    state.runtimeReady = false;
  }
  if (next.percent != null) {
    runtimeState.percent = next.percent;
  }
  runtimeState.detail = next.detail;
  runtimeState.lastUpdateAt = Date.now();
  pushRuntimeLog(next.logText, next.logKey);
  setStatus(nextStatus);
  applyInteractivity();
  renderRuntimePanel();
}

function flushStatus() {
  if (statusState.timer) {
    clearTimeout(statusState.timer);
    statusState.timer = null;
  }
  if (statusState.pending == null || statusState.pending === statusState.lastRendered) {
    return;
  }
  statusState.lastRendered = statusState.pending;
  renderRuntimePanel();
}

function setStatus(message, { immediate = false } = {}) {
  statusState.pending = message;
  if (immediate) {
    flushStatus();
    return;
  }
  if (statusState.timer) {
    return;
  }
  statusState.timer = setTimeout(flushStatus, 120);
}

function selectedModelIsReady() {
  const modelId = elements.modelId.value.trim();
  return Boolean(modelId && state.loadedModelId === modelId && state.runtimeReady);
}

function applyInteractivity() {
  const disabled = state.loading || state.generating;
  const modelSelectionDisabled = state.generating;
  const promptLocked = disabled || !selectedModelIsReady();

  elements.loadModel.disabled = disabled;
  elements.clearChat.disabled = disabled;
  elements.clearCache.disabled = disabled;
  elements.sendMessage.disabled = promptLocked;
  elements.promptInput.disabled = promptLocked;
  elements.modelId.disabled = modelSelectionDisabled;
  elements.presetModel.disabled = modelSelectionDisabled;
  elements.hfToken.disabled = disabled;
  elements.imageInput.disabled = disabled;
  elements.audioInput.disabled = disabled || !selectedPresetSupports("audio");
  elements.videoInput.disabled = disabled || !selectedPresetSupports("video");
  elements.systemPrompt.disabled = disabled;
  elements.maxTokens.disabled = disabled;
  elements.temperature.disabled = disabled;
}

function setBusy({ loading = state.loading, generating = state.generating } = {}) {
  state.loading = loading;
  state.generating = generating;
  applyInteractivity();
}

function hasComposerMedia() {
  return Boolean(state.composerMedia.image || state.composerMedia.audio || state.composerMedia.video);
}

function makeReadyStatus(modelId, textOnly) {
  if (!modelId) {
    return "Ready.";
  }
  return textOnly
    ? `Ready: ${modelId} (text sessions warm)`
    : `Ready: ${modelId} (multimodal sessions warm)`;
}

function currentPendingDots() {
  return ".".repeat((pendingDotsState.frame % 3) + 1);
}

function refreshPendingDots() {
  for (const node of document.querySelectorAll(".bubble-body-pending")) {
    node.textContent = currentPendingDots();
  }
}

function syncPendingDotsTimer() {
  const shouldAnimate = state.messages.some((message) => message.pending && !message.content);
  if (shouldAnimate && !pendingDotsState.timer) {
    pendingDotsState.timer = setInterval(() => {
      pendingDotsState.frame = (pendingDotsState.frame + 1) % 3;
      refreshPendingDots();
    }, 360);
    return;
  }

  if (!shouldAnimate && pendingDotsState.timer) {
    clearInterval(pendingDotsState.timer);
    pendingDotsState.timer = null;
    pendingDotsState.frame = 0;
  }
}

function truncateContext(messages, turns = 10) {
  const maxMessages = turns * 2;
  return messages.slice(-maxMessages);
}

function createBubble(message) {
  const wrapper = document.createElement("article");
  wrapper.className = `bubble bubble-${message.role}`;

  const label = document.createElement("p");
  label.className = "bubble-label";
  label.textContent = message.role === "assistant" ? "assistant" : "user";

  const body = document.createElement("p");
  body.className = "bubble-body";
  if (message.pending && !message.content) {
    body.classList.add("bubble-body-pending");
    body.textContent = currentPendingDots();
  } else {
    body.textContent = message.content || "";
  }

  wrapper.append(label);

  const attachments = [];
  if (message.image) {
    attachments.push(`image: ${message.image.name}`);
  }
  if (message.audio) {
    attachments.push(`audio: ${message.audio.name}`);
  }
  if (message.video) {
    attachments.push(`video: ${message.video.name}`);
  }

  if (attachments.length > 0) {
    const list = document.createElement("div");
    list.className = "attachment-list";
    for (const entry of attachments) {
      const chip = document.createElement("span");
      chip.className = "attachment-chip";
      chip.textContent = entry;
      list.append(chip);
    }
    wrapper.append(list);
  }

  wrapper.append(body);
  return wrapper;
}

function renderMessages() {
  elements.messages.innerHTML = "";

  if (state.messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent =
      "Choose a preset, load it, then chat or attach media. The UI stays responsive while the browser runtime downloads and compiles in the background.";
    elements.messages.append(empty);
    return;
  }

  for (const message of state.messages) {
    elements.messages.append(createBubble(message));
  }
  syncPendingDotsTimer();
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function syncPresetSelection() {
  const selected = findModelPreset(elements.modelId.value.trim());
  elements.presetModel.value = selected?.modelId || "__custom__";
  elements.presetNote.textContent = formatPresetSummary(selected);
  syncComposerMediaState();
  applyInteractivity();
}

function selectedPresetSupports(modality) {
  return Boolean(findModelPreset(elements.modelId.value.trim())?.modalities?.includes(modality));
}

function revokeMediaEntry(entry) {
  if (entry?.url) {
    URL.revokeObjectURL(entry.url);
  }
}

function resetComposerMedia({ revoke = true } = {}) {
  if (revoke) {
    revokeMediaEntry(state.composerMedia.image);
    revokeMediaEntry(state.composerMedia.audio);
    revokeMediaEntry(state.composerMedia.video);
  }
  state.composerMedia.image = null;
  state.composerMedia.audio = null;
  state.composerMedia.video = null;
  elements.imageInput.value = "";
  elements.audioInput.value = "";
  elements.videoInput.value = "";
  syncComposerMediaState();
}

function revokeMessageMedia(messages) {
  for (const message of messages) {
    revokeMediaEntry(message.image);
    revokeMediaEntry(message.audio);
    revokeMediaEntry(message.video);
  }
}

function describeMediaEntry(entry, fallback) {
  if (!entry) {
    return fallback;
  }
  const size = entry.file?.size ? ` | ${(entry.file.size / (1024 * 1024)).toFixed(1)} MB` : "";
  return `${entry.name}${size}`;
}

function syncComposerMediaState() {
  elements.imageStatus.textContent = describeMediaEntry(
    state.composerMedia.image,
    selectedPresetSupports("image") ? "No image selected." : "This preset does not expose image input.",
  );

  const audioEnabled = selectedPresetSupports("audio");
  if (!audioEnabled && state.composerMedia.audio) {
    revokeMediaEntry(state.composerMedia.audio);
    state.composerMedia.audio = null;
    elements.audioInput.value = "";
  }
  elements.audioInput.disabled = state.loading || state.generating || !audioEnabled;
  elements.audioStatus.textContent = describeMediaEntry(
    state.composerMedia.audio,
    audioEnabled ? "No audio selected." : "Audio input is enabled only for presets that ship audio support.",
  );

  const videoEnabled = selectedPresetSupports("video");
  if (!videoEnabled && state.composerMedia.video) {
    revokeMediaEntry(state.composerMedia.video);
    state.composerMedia.video = null;
    elements.videoInput.value = "";
  }
  elements.videoInput.disabled = state.loading || state.generating || !videoEnabled;
  elements.videoStatus.textContent = describeMediaEntry(
    state.composerMedia.video,
    videoEnabled ? "No video selected." : "Video input is enabled only for presets that ship video support.",
  );
}

function buildConversationMessages(messages) {
  return messages.map((message) => {
    if (message.role !== "user" || (!message.image && !message.audio && !message.video)) {
      return {
        role: message.role,
        content: message.content,
      };
    }

    const content = [];
    if (message.image) {
      content.push({ type: "image", url: message.image.url });
    }
    if (message.audio) {
      content.push({ type: "audio", url: message.audio.url });
    }
    if (message.video) {
      content.push({ type: "video", url: message.video.url });
    }
    if (typeof message.content === "string" && message.content.trim()) {
      content.push({ type: "text", text: message.content.trim() });
    }

    return {
      role: message.role,
      content,
    };
  });
}

function defaultPromptForComposerMedia() {
  if (state.composerMedia.image && state.composerMedia.audio) {
    return "Describe the attached image and transcribe the attached audio.";
  }
  if (state.composerMedia.image && state.composerMedia.video) {
    return "Describe the attached image and summarize the attached video.";
  }
  if (state.composerMedia.audio && state.composerMedia.video) {
    return "Transcribe the attached audio and summarize the attached video.";
  }
  if (state.composerMedia.image) {
    return "Describe the attached image.";
  }
  if (state.composerMedia.audio) {
    return "Transcribe the attached audio.";
  }
  if (state.composerMedia.video) {
    return "Summarize the attached video.";
  }
  return "";
}

function createMediaEntry(file) {
  return {
    file,
    name: file.name,
    type: file.type,
    url: URL.createObjectURL(file),
  };
}

function handleMediaSelection(kind, file) {
  revokeMediaEntry(state.composerMedia[kind]);
  state.composerMedia[kind] = file ? createMediaEntry(file) : null;
  syncComposerMediaState();
}

function setQueryModel(modelId) {
  const url = new URL(window.location.href);
  url.searchParams.set("model", modelId);
  window.history.replaceState({}, "", url);
}

async function loadModel() {
  const modelId = elements.modelId.value.trim();
  const preset = findModelPreset(modelId);
  if (!modelId) {
    setStatus("Model ID is required.");
    return;
  }

  try {
    setBusy({ loading: true, generating: false });
    beginRuntimeActivity(`Loading text sessions for ${modelId}...`, {
      detail: "Preparing text-only session warm load.",
      resetProgress: true,
    });
    await initializeLocalAuthToken();
    const result = await state.runtime.load({
      modelId,
      modelFamily: preset?.family,
      dtype: preset?.dtype,
      authToken: getAuthToken(),
      textOnly: true,
      onProgress: handleRuntimeProgress,
    });
    state.loadedModelId = modelId;
    state.loadedTextOnly = result?.textOnly ?? true;
    setQueryModel(modelId);
    finishRuntimeActivity(result?.readyMessage ?? makeReadyStatus(modelId, state.loadedTextOnly), {
      detail: result?.readyMessage ?? "Text sessions are ready.",
    });
  } catch (error) {
    failRuntimeActivity(`Load failed: ${formatRuntimeError(error)}`);
  } finally {
    setBusy({ loading: false, generating: false });
  }
}

async function sendMessage() {
  const typedPrompt = elements.promptInput.value.trim();
  const prompt = typedPrompt || defaultPromptForComposerMedia();
  const needsMultimodal = hasComposerMedia();
  if (!prompt && !needsMultimodal) {
    return;
  }

  const modelId = elements.modelId.value.trim();
  const preset = findModelPreset(modelId);

  const userMessage = {
    role: "user",
    content: prompt,
    image: state.composerMedia.image,
    audio: state.composerMedia.audio,
    video: state.composerMedia.video,
  };
  const assistantMessage = { role: "assistant", content: "", pending: true };
  state.messages.push(userMessage, assistantMessage);
  renderMessages();
  elements.promptInput.value = "";
  state.composerMedia.image = null;
  state.composerMedia.audio = null;
  state.composerMedia.video = null;
  elements.imageInput.value = "";
  elements.audioInput.value = "";
  elements.videoInput.value = "";
  syncComposerMediaState();

  try {
    setBusy({ loading: false, generating: true });
    const preparingStatus =
      !state.loadedModelId || state.loadedModelId !== modelId
        ? needsMultimodal
          ? `Loading multimodal sessions for ${modelId}...`
          : `Loading text sessions for ${modelId}...`
        : state.loadedTextOnly && needsMultimodal
          ? `Upgrading ${modelId} to multimodal sessions...`
          : `Generating with ${modelId}...`;
    beginRuntimeActivity(preparingStatus, {
      detail: needsMultimodal
        ? "Preparing multimodal runtime sessions."
        : "Preparing text generation.",
      resetProgress:
        !state.loadedModelId || state.loadedModelId !== modelId || (state.loadedTextOnly && needsMultimodal),
    });

    await initializeLocalAuthToken();
    const result = await state.runtime.generate({
      modelId,
      modelFamily: preset?.family,
      dtype: preset?.dtype,
      authToken: getAuthToken(),
      systemPrompt: elements.systemPrompt.value,
      messages: buildConversationMessages(
        truncateContext(state.messages.filter((message) => !message.pending)),
      ),
      maxNewTokens: Number(elements.maxTokens.value),
      temperature: Number(elements.temperature.value),
      onProgress: handleRuntimeProgress,
      onToken: (fullText) => {
        assistantMessage.content = fullText;
        renderMessages();
      },
    });

    assistantMessage.content = result || assistantMessage.content || "No output returned.";
    assistantMessage.pending = false;
    state.loadedModelId = modelId;
    state.loadedTextOnly = !needsMultimodal;
    setQueryModel(modelId);
    finishRuntimeActivity(makeReadyStatus(modelId, state.loadedTextOnly), {
      detail: "Generation complete.",
      percent: runtimeState.percent ?? 100,
    });
  } catch (error) {
    state.messages.pop();
    failRuntimeActivity(`Generation failed: ${formatRuntimeError(error)}`);
  } finally {
    setBusy({ loading: false, generating: false });
    renderMessages();
  }
}

function clearChat() {
  revokeMessageMedia(state.messages);
  resetComposerMedia();
  state.messages = [];
  renderMessages();
  state.runtimeReady = Boolean(state.loadedModelId);
  setStatus(state.loadedModelId ? `Ready: ${state.loadedModelId}` : "Chat cleared.", { immediate: true });
  renderRuntimePanel();
}

async function clearCache() {
  try {
    setBusy({ loading: true, generating: false });
    beginRuntimeActivity("Clearing browser model cache...", {
      detail: "Removing cached ONNX/browser artifacts.",
      resetProgress: true,
    });
    const result = await state.runtime.clearCache();
    state.runtimeReady = false;
    state.loadedModelId = null;
    state.loadedTextOnly = null;
    if (!result.supported) {
      failRuntimeActivity("Browser cache API is not available in this environment.");
      return;
    }
    finishRuntimeActivity(
      result.deleted.length > 0
        ? `Cleared ${result.deleted.length} browser cache entr${result.deleted.length === 1 ? "y" : "ies"}.`
        : "No cached Transformers.js model files were found.",
      {
        detail:
          result.deleted.length > 0
            ? `Removed ${result.deleted.length} cached browser entries.`
            : "No cached browser entries were present.",
        percent: null,
      },
    );
  } catch (error) {
    failRuntimeActivity(`Cache clear failed: ${formatRuntimeError(error)}`);
  } finally {
    setBusy({ loading: false, generating: false });
  }
}

function initializePresets() {
  const fragment = document.createDocumentFragment();
  for (const preset of DEFAULT_MODEL_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.modelId;
    option.textContent = preset.label;
    fragment.append(option);
  }

  const customOption = document.createElement("option");
  customOption.value = "__custom__";
  customOption.textContent = "Custom model ID";
  fragment.append(customOption);

  elements.presetModel.append(fragment);
  elements.modelId.value = initialModelId();
  syncPresetSelection();
}

function bindEvents() {
  elements.presetModel.addEventListener("change", () => {
    if (elements.presetModel.value === "__custom__") {
      return;
    }
    elements.modelId.value = elements.presetModel.value;
    syncPresetSelection();
  });

  elements.modelId.addEventListener("input", () => {
    syncPresetSelection();
  });

  elements.hfToken.addEventListener("input", () => {
    const token = getAuthToken();
    if (token) {
      window.localStorage.setItem(HF_TOKEN_STORAGE_KEY, token);
    } else {
      window.localStorage.removeItem(HF_TOKEN_STORAGE_KEY);
    }
  });

  elements.imageInput.addEventListener("change", () => {
    handleMediaSelection("image", elements.imageInput.files?.[0] ?? null);
  });

  elements.audioInput.addEventListener("change", () => {
    handleMediaSelection("audio", elements.audioInput.files?.[0] ?? null);
  });

  elements.videoInput.addEventListener("change", () => {
    handleMediaSelection("video", elements.videoInput.files?.[0] ?? null);
  });

  elements.loadModel.addEventListener("click", (event) => {
    event.preventDefault();
    loadModel();
  });

  elements.clearChat.addEventListener("click", (event) => {
    event.preventDefault();
    clearChat();
  });

  elements.clearCache.addEventListener("click", async (event) => {
    event.preventDefault();
    await clearCache();
  });

  elements.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendMessage();
  });

  elements.promptInput.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      await sendMessage();
    }
  });
}

function initializeRuntimeStatus() {
  if (!navigator.gpu) {
    runtimeState.detail = "WebGPU is unavailable in this browser.";
    setStatus("WebGPU is not available. Use a modern desktop browser with WebGPU enabled.", { immediate: true });
    setBusy({ loading: false, generating: false });
    elements.loadModel.disabled = true;
    elements.sendMessage.disabled = true;
    return;
  }
  runtimeState.detail = "Waiting for model activity.";
  setStatus("WebGPU detected. Load a model to start.", { immediate: true });
}

initializeAuthToken();
initializePresets();
bindEvents();
renderMessages();
initializeRuntimeStatus();
syncComposerMediaState();
renderRuntimeLog();
renderRuntimePanel();
initializeLocalAuthToken();
applyInteractivity();
setInterval(() => {
  if (runtimeState.lastUpdateAt) {
    renderRuntimeAge();
  }
}, 1000);
