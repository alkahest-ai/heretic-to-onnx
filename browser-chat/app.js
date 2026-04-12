import {
  clearBrowserModelCache,
  createBrowserChatRuntime,
  DEFAULT_MODEL_ID,
  DEFAULT_MODEL_PRESETS,
  findModelPreset,
  formatPresetSummary,
} from "../examples/browser-loader.mjs";

const elements = {
  presetModel: document.querySelector("#preset-model"),
  modelId: document.querySelector("#model-id"),
  presetNote: document.querySelector("#preset-note"),
  imageInput: document.querySelector("#image-input"),
  imageStatus: document.querySelector("#image-status"),
  audioInput: document.querySelector("#audio-input"),
  audioStatus: document.querySelector("#audio-status"),
  systemPrompt: document.querySelector("#system-prompt"),
  maxTokens: document.querySelector("#max-tokens"),
  temperature: document.querySelector("#temperature"),
  loadModel: document.querySelector("#load-model"),
  clearChat: document.querySelector("#clear-chat"),
  clearCache: document.querySelector("#clear-cache"),
  runtimeStatus: document.querySelector("#runtime-status"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chat-form"),
  promptInput: document.querySelector("#prompt-input"),
  sendMessage: document.querySelector("#send-message"),
};

const state = {
  runtime: createBrowserChatRuntime(),
  messages: [],
  loading: false,
  generating: false,
  loadedModelId: null,
  composerMedia: {
    image: null,
    audio: null,
  },
};

function initialModelId() {
  const url = new URL(window.location.href);
  return url.searchParams.get("model") || DEFAULT_MODEL_ID;
}

function setStatus(message) {
  elements.runtimeStatus.textContent = message;
}

function setBusy({ loading = state.loading, generating = state.generating } = {}) {
  state.loading = loading;
  state.generating = generating;
  const disabled = state.loading || state.generating;

  elements.loadModel.disabled = disabled;
  elements.clearChat.disabled = disabled;
  elements.clearCache.disabled = disabled;
  elements.sendMessage.disabled = disabled;
  elements.promptInput.disabled = disabled;
  elements.modelId.disabled = disabled;
  elements.presetModel.disabled = disabled;
  elements.imageInput.disabled = disabled;
  elements.audioInput.disabled = disabled || !selectedPresetSupportsAudio();
  elements.systemPrompt.disabled = disabled;
  elements.maxTokens.disabled = disabled;
  elements.temperature.disabled = disabled;
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
  label.textContent = message.role === "assistant" ? "Model" : "User";

  const body = document.createElement("p");
  body.className = "bubble-body";
  body.textContent = message.content || (message.pending ? "..." : "");

  wrapper.append(label);

  const attachments = [];
  if (message.image) {
    attachments.push(`image: ${message.image.name}`);
  }
  if (message.audio) {
    attachments.push(`audio: ${message.audio.name}`);
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
      "Load a model, then send a prompt. The response is generated locally in the browser with WebGPU.";
    elements.messages.append(empty);
    return;
  }

  for (const message of state.messages) {
    elements.messages.append(createBubble(message));
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function syncPresetSelection() {
  const selected = findModelPreset(elements.modelId.value.trim());
  elements.presetModel.value = selected?.modelId || "__custom__";
  elements.presetNote.textContent = formatPresetSummary(selected);
  syncComposerMediaState();
}

function selectedPresetSupportsAudio() {
  return findModelPreset(elements.modelId.value.trim())?.family === "gemma4";
}

function selectedPresetSupportsImage() {
  return Boolean(findModelPreset(elements.modelId.value.trim())?.modalities?.includes("image"));
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
  }
  state.composerMedia.image = null;
  state.composerMedia.audio = null;
  elements.imageInput.value = "";
  elements.audioInput.value = "";
  syncComposerMediaState();
}

function revokeMessageMedia(messages) {
  for (const message of messages) {
    revokeMediaEntry(message.image);
    revokeMediaEntry(message.audio);
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
    selectedPresetSupportsImage() ? "No image selected." : "This preset does not expose image input.",
  );

  const audioEnabled = selectedPresetSupportsAudio();
  if (!audioEnabled && state.composerMedia.audio) {
    revokeMediaEntry(state.composerMedia.audio);
    state.composerMedia.audio = null;
    elements.audioInput.value = "";
  }
  elements.audioInput.disabled = state.loading || state.generating || !audioEnabled;
  elements.audioStatus.textContent = describeMediaEntry(
    state.composerMedia.audio,
    audioEnabled ? "No audio selected." : "Audio input is enabled only for Rally / Gemma presets.",
  );
}

function buildConversationMessages(messages) {
  return messages.map((message) => {
    if (message.role !== "user" || (!message.image && !message.audio)) {
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
  if (state.composerMedia.image) {
    return "Describe the attached image.";
  }
  if (state.composerMedia.audio) {
    return "Transcribe the attached audio.";
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
    setStatus(`Loading ${modelId}...`);
    await state.runtime.load({
      modelId,
      modelFamily: preset?.family,
      dtype: preset?.dtype,
      onProgress: (_info, message) => setStatus(message),
    });
    state.loadedModelId = modelId;
    setQueryModel(modelId);
    setStatus(`Ready: ${modelId}`);
  } catch (error) {
    setStatus(`Load failed: ${error.message}`);
  } finally {
    setBusy({ loading: false, generating: false });
  }
}

async function sendMessage() {
  const typedPrompt = elements.promptInput.value.trim();
  const prompt = typedPrompt || defaultPromptForComposerMedia();
  if (!prompt && !state.composerMedia.image && !state.composerMedia.audio) {
    return;
  }

  const modelId = elements.modelId.value.trim();
  const preset = findModelPreset(modelId);
  if (!state.loadedModelId || state.loadedModelId !== modelId) {
    await loadModel();
    if (state.loadedModelId !== modelId) {
      return;
    }
  }

  const userMessage = {
    role: "user",
    content: prompt,
    image: state.composerMedia.image,
    audio: state.composerMedia.audio,
  };
  const assistantMessage = { role: "assistant", content: "", pending: true };
  state.messages.push(userMessage, assistantMessage);
  renderMessages();
  elements.promptInput.value = "";
  state.composerMedia.image = null;
  state.composerMedia.audio = null;
  elements.imageInput.value = "";
  elements.audioInput.value = "";
  syncComposerMediaState();

  try {
    setBusy({ loading: false, generating: true });
    setStatus(`Generating with ${modelId}...`);

    const result = await state.runtime.generate({
      modelId,
      modelFamily: preset?.family,
      dtype: preset?.dtype,
      systemPrompt: elements.systemPrompt.value,
      messages: buildConversationMessages(
        truncateContext(state.messages.filter((message) => !message.pending)),
      ),
      maxNewTokens: Number(elements.maxTokens.value),
      temperature: Number(elements.temperature.value),
      onProgress: (_info, message) => setStatus(message),
      onToken: (fullText) => {
        assistantMessage.content = fullText;
        renderMessages();
      },
    });

    assistantMessage.content = result || assistantMessage.content || "No output returned.";
    assistantMessage.pending = false;
    setStatus(`Ready: ${modelId}`);
  } catch (error) {
    state.messages.pop();
    setStatus(`Generation failed: ${error.message}`);
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
  setStatus(state.loadedModelId ? `Ready: ${state.loadedModelId}` : "Chat cleared.");
}

async function clearCache() {
  try {
    setBusy({ loading: true, generating: false });
    setStatus("Clearing browser model cache...");
    await state.runtime.dispose();
    const result = await clearBrowserModelCache();
    state.loadedModelId = null;
    if (!result.supported) {
      setStatus("Browser cache API is not available in this environment.");
      return;
    }
    setStatus(
      result.deleted.length > 0
        ? `Cleared ${result.deleted.length} browser cache entr${result.deleted.length === 1 ? "y" : "ies"}.`
        : "No cached Transformers.js model files were found.",
    );
  } catch (error) {
    setStatus(`Cache clear failed: ${error.message}`);
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

  elements.imageInput.addEventListener("change", () => {
    handleMediaSelection("image", elements.imageInput.files?.[0] ?? null);
  });

  elements.audioInput.addEventListener("change", () => {
    handleMediaSelection("audio", elements.audioInput.files?.[0] ?? null);
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
    setStatus("WebGPU is not available. Use a modern desktop browser with WebGPU enabled.");
    setBusy({ loading: false, generating: false });
    elements.loadModel.disabled = true;
    elements.sendMessage.disabled = true;
    return;
  }
  setStatus("WebGPU detected. Load a model to start.");
}

initializePresets();
bindEvents();
renderMessages();
initializeRuntimeStatus();
syncComposerMediaState();
