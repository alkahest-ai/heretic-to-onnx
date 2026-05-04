import * as Transformers from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const { AutoConfig, AutoModelForVision2Seq, AutoProcessor, TextStreamer, env, load_image, read_audio } =
  Transformers;

env.allowLocalModels = true;
env.allowRemoteModels = true;
env.useBrowserCache = true;

const defaultFetch = globalThis.fetch?.bind(globalThis);

function configureHubAuth(authToken) {
  if (!defaultFetch) {
    return;
  }

  if (!authToken) {
    env.fetch = defaultFetch;
    if (env.HF_TOKEN) {
      delete env.HF_TOKEN;
    }
    return;
  }

  env.HF_TOKEN = authToken;
  env.fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : input?.url;
    const nextInit = { ...init };
    if (typeof url === "string") {
      const parsed = new URL(url, globalThis.location?.href);
      if (parsed.hostname === "huggingface.co") {
        const headers = new Headers(init.headers ?? (input instanceof Request ? input.headers : undefined));
        headers.set("Authorization", `Bearer ${authToken}`);
        nextInit.headers = headers;
      }
    }
    return defaultFetch(input, nextInit);
  };
}

export const PUBLIC_MODEL_OWNER = "thomasjvu";

function ownedModel(modelName) {
  return `${PUBLIC_MODEL_OWNER}/${modelName}`;
}

const QWEN35_WEBGPU_DTYPE = Object.freeze({
  embed_tokens: "q4",
  decoder_model_merged: "q4",
  vision_encoder: "fp16",
});

const QWEN35_WEBGPU_TEXT_DTYPE = Object.freeze({
  embed_tokens: "q4",
  decoder_model_merged: "q4",
});

const QWEN35_WEBGPU_Q4_VISION_DTYPE = Object.freeze({
  embed_tokens: "q4",
  decoder_model_merged: "q4",
  vision_encoder: "q4",
});

const QWEN35_WEBGPU_Q8_DTYPE = Object.freeze({
  embed_tokens: "q8",
  decoder_model_merged: "q8",
  vision_encoder: "q8",
});

const GEMMA4_WEBGPU_DTYPE = Object.freeze({
  embed_tokens: "q4f16",
  decoder_model_merged: "q4f16",
  vision_encoder: "q4f16",
});

const GEMMA4_WEBGPU_TEXT_DTYPE = Object.freeze({
  embed_tokens: "q4f16",
  decoder_model_merged: "q4f16",
});

export const DEFAULT_MODEL_ID = ownedModel("alkahest-0.8b-heretic-q4-onnx");

export const DEFAULT_MODEL_PRESETS = [
  {
    label: "Alkahest 0.8B Heretic Q4 (stable)",
    modelId: ownedModel("alkahest-0.8b-heretic-q4-onnx"),
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~850 MB",
    dtype: QWEN35_WEBGPU_DTYPE,
    note: "Stable Heretic-only 0.8B q4 browser package. Use this as the baseline for SFT comparisons.",
  },
  {
    label: "Alkahest 0.8B Heretic Q4 Text",
    modelId: ownedModel("alkahest-0.8b-heretic-q4-onnx-text"),
    family: "qwen3_5",
    modalities: "text",
    approxDownload: "~620 MB",
    dtype: QWEN35_WEBGPU_TEXT_DTYPE,
    note: "Text-only 0.8B Heretic q4 package for the lightest Alkahest browser smoke target.",
  },
  {
    label: "Alkahest 0.8B Heretic RP v8 A50/B100 Q4",
    modelId: ownedModel("alkahest-0.8b-heretic-rp-sft-two-stage-a50-b100-q4-onnx"),
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~850 MB",
    dtype: QWEN35_WEBGPU_DTYPE,
    note: "Promoted 0.8B RP candidate. Browser scorecard total 0.8500 with a +0.3225 margin over direct.",
  },
  {
    label: "Alkahest 0.8B Heretic RP v8 A25/B100 Q4",
    modelId: ownedModel("alkahest-0.8b-heretic-rp-sft-two-stage-a25-b100-q4-onnx"),
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~850 MB",
    dtype: QWEN35_WEBGPU_DTYPE,
    note: "Softer promoted 0.8B RP candidate. Browser scorecard total 0.7625 with a +0.2350 margin over direct.",
  },
  {
    label: "Alkahest 2B Heretic Q4",
    modelId: ownedModel("alkahest-2b-heretic-q4-onnx"),
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~2.1 GB",
    dtype: QWEN35_WEBGPU_DTYPE,
    note: "Full multimodal 2B Heretic q4 package. Larger than the text-only build because it keeps the fp16 vision encoder.",
  },
  {
    label: "Alkahest 2B Heretic Q4 Text",
    modelId: ownedModel("alkahest-2b-heretic-q4-onnx-text"),
    family: "qwen3_5",
    modalities: "text",
    approxDownload: "~1.45 GB",
    dtype: QWEN35_WEBGPU_TEXT_DTYPE,
    note: "Text-only 2B Heretic q4 package. This avoids loading the vision encoder for browser chat.",
  },
  {
    label: "Alkahest 2B Heretic RP v8 A100/B75 Q4",
    modelId: ownedModel("alkahest-2b-heretic-rp-sft-two-stage-a100-b75-q4-onnx"),
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~2.1 GB",
    dtype: QWEN35_WEBGPU_DTYPE,
    note: "Promoted 2B RP candidate. Browser scorecard total 0.8025 with a +0.2350 margin over direct.",
  },
];

function normalizeMessages(messages) {
  const promptMessages = [];
  const imageSources = [];
  const audioSources = [];
  const videoSources = [];

  for (const message of messages ?? []) {
    if (!message?.role) {
      continue;
    }

    if (typeof message.content === "string") {
      const text = message.content.trim();
      if (!text) {
        continue;
      }
      promptMessages.push({
        role: message.role,
        content: text,
      });
      continue;
    }

    if (!Array.isArray(message.content)) {
      continue;
    }

    const content = [];
    for (const item of message.content) {
      if (item?.type === "text") {
        const text = typeof item.text === "string" ? item.text.trim() : "";
        if (text) {
          content.push({ type: "text", text });
        }
        continue;
      }

      if (item?.type === "image") {
        content.push({ type: "image" });
        if (typeof item.url === "string" && item.url) {
          imageSources.push(item.url);
        }
        continue;
      }

      if (item?.type === "audio") {
        content.push({ type: "audio" });
        if (typeof item.url === "string" && item.url) {
          audioSources.push(item.url);
        }
        continue;
      }

      if (item?.type === "video") {
        content.push({ type: "video" });
        if (typeof item.url === "string" && item.url) {
          videoSources.push(item.url);
        }
      }
    }

    if (content.length > 0) {
      promptMessages.push({
        role: message.role,
        content,
      });
    }
  }

  return { promptMessages, imageSources, audioSources, videoSources };
}

function makeProgressMessage(info) {
  if (!info || typeof info !== "object") {
    return "Loading model...";
  }
  if (info.status === "progress_total" && typeof info.progress === "number") {
    const percent = Math.round(info.progress);
    return info.file ? `Loading ${info.file} (${percent}%)` : `Loading model (${percent}%)`;
  }
  if (info.status === "done") {
    return "Download complete. Initializing WebGPU sessions...";
  }
  if (typeof info.status === "string") {
    return info.file ? `${info.status}: ${info.file}` : info.status;
  }
  return "Loading model...";
}

function dtypeUsesBrowserFloat16Metadata(dtype) {
  if (typeof dtype === "string") {
    return ["q4f16", "fp16", "q4"].includes(dtype);
  }
  if (dtype && typeof dtype === "object") {
    return Object.values(dtype).some((value) => ["q4f16", "fp16", "q4"].includes(value));
  }
  return false;
}

function sanitizeBrowserConfig(config, dtype) {
  if (!config || typeof config !== "object") {
    return config;
  }

  if (!dtypeUsesBrowserFloat16Metadata(dtype)) {
    return config;
  }

  const replaceNode = (node) => {
    if (Array.isArray(node)) {
      return node.map(replaceNode);
    }
    if (node && typeof node === "object") {
      const next = {};
      for (const [key, value] of Object.entries(node)) {
        if ((key === "dtype" || key === "torch_dtype") && value === "bfloat16") {
          next[key] = "float16";
        } else {
          next[key] = replaceNode(value);
        }
      }
      return next;
    }
    return node;
  };

  return replaceNode(config);
}

function resolveModelClass(family, { textOnly = false } = {}) {
  const candidates =
    family === "gemma4"
      ? textOnly
        ? [Transformers.Gemma4ForCausalLM, Transformers.Gemma4ForConditionalGeneration]
        : [Transformers.Gemma4ForConditionalGeneration]
      : textOnly
        ? [
            Transformers.Qwen3_5ForCausalLM,
            Transformers.Qwen3ForCausalLM,
            Transformers.Qwen3_5ForConditionalGeneration,
            Transformers.Qwen3VLForConditionalGeneration,
            Transformers.Qwen2_5_VLForConditionalGeneration,
          ]
        : [
            Transformers.Qwen3_5ForConditionalGeneration,
            Transformers.Qwen3VLForConditionalGeneration,
            Transformers.Qwen2_5_VLForConditionalGeneration,
          ];

  for (const candidate of candidates) {
    if (typeof candidate?.from_pretrained === "function") {
      return candidate;
    }
  }
  if (typeof AutoModelForVision2Seq?.from_pretrained === "function") {
    return AutoModelForVision2Seq;
  }
  throw new Error(`No compatible Transformers.js class is available for ${family}.`);
}

function resolveProcessorClass(family) {
  const candidates =
    family === "gemma4"
      ? [Transformers.Gemma4Processor, AutoProcessor]
      : [Transformers.Qwen3VLProcessor, Transformers.Qwen2_5_VLProcessor, AutoProcessor];

  for (const candidate of candidates) {
    if (typeof candidate?.from_pretrained === "function") {
      return candidate;
    }
  }

  throw new Error(`No compatible Transformers.js processor is available for ${family}.`);
}

async function buildProcessorInputs(processor, family, prompt, imageArg, audioArg, videoArg) {
  const options = { add_special_tokens: false };
  if (!imageArg && !audioArg && !videoArg) {
    return family === "qwen3_5" ? processor(prompt) : processor(prompt, options);
  }

  const hasTensorValues = (value) => {
    if (!value) {
      return false;
    }
    if (Array.isArray(value)) {
      return value.length > 0;
    }
    if (ArrayBuffer.isView(value)) {
      return value.length > 0;
    }
    if (value.data && ArrayBuffer.isView(value.data)) {
      return value.data.length > 0;
    }
    if (Array.isArray(value.dims)) {
      return value.dims.every((dim) => Number(dim) > 0);
    }
    return false;
  };

  const validateMultimodalInputs = (inputs) => {
    if (family !== "qwen3_5") {
      return inputs;
    }
    if (imageArg && !hasTensorValues(inputs?.image_grid_thw)) {
      throw new Error("Qwen processor output omitted image_grid_thw for image input.");
    }
    if (videoArg && !hasTensorValues(inputs?.video_grid_thw)) {
      throw new Error("Qwen processor output omitted video_grid_thw for video input.");
    }
    return inputs;
  };

  const attempt = async (factory) => {
    try {
      return validateMultimodalInputs(await factory());
    } catch (error) {
      return error;
    }
  };

  const attempts = [
    () =>
      processor({
        text: prompt,
        images: imageArg,
        audio: audioArg,
        videos: videoArg,
        ...options,
      }),
    () =>
      processor({
        text: prompt,
        images: imageArg,
        audios: audioArg,
        videos: videoArg,
        ...options,
      }),
  ];

  if (family === "gemma4") {
    attempts.push(() => processor(prompt, imageArg, audioArg, videoArg, options));
    attempts.push(() => processor(prompt, imageArg, audioArg, { videos: videoArg, ...options }));
    attempts.push(() => processor(prompt, imageArg, audioArg, options));
  } else {
    attempts.push(() => processor(prompt, imageArg, videoArg, options));
    attempts.push(() => processor(prompt, imageArg, { videos: videoArg, ...options }));
    attempts.push(() => processor(prompt, imageArg, options));
  }

  let lastError = null;
  for (const factory of attempts) {
    const result = await attempt(factory);
    if (!(result instanceof Error)) {
      return result;
    }
    lastError = result;
  }

  throw lastError ?? new Error("unable to build multimodal processor inputs");
}

function firstDecodedText(processor, outputs) {
  const decoded = processor.batch_decode(outputs, {
    skip_special_tokens: true,
  });
  return typeof decoded?.[0] === "string" ? decoded[0] : "";
}

function decodeGeneratedText(processor, outputs, inputs, streamedText) {
  const streamed = streamedText.trim();
  if (streamed) {
    return streamed;
  }

  const promptTokens = Number(inputs?.input_ids?.dims?.at?.(-1));
  if (Number.isFinite(promptTokens) && promptTokens > 0 && typeof outputs?.slice === "function") {
    try {
      const generatedOnly = outputs.slice(null, [promptTokens, null]);
      const decoded = firstDecodedText(processor, generatedOnly).trim();
      if (decoded) {
        return decoded;
      }
    } catch {
      // Tensor slicing differs across Transformers.js releases; full decode is safer than failing a completed generation.
    }
  }

  const decoded = firstDecodedText(processor, outputs).trim();
  if (!decoded || !inputs?.input_ids) {
    return decoded;
  }

  try {
    const prompt = firstDecodedText(processor, inputs.input_ids).trim();
    return prompt && decoded.startsWith(prompt) ? decoded.slice(prompt.length).trim() : decoded;
  } catch {
    return decoded;
  }
}

export function findModelPreset(modelId) {
  return DEFAULT_MODEL_PRESETS.find((preset) => preset.modelId === modelId) ?? null;
}

export function inferModelFamily(modelId) {
  const value = String(modelId || "").toLowerCase();
  if (value.includes("alkahest")) {
    return "qwen3_5";
  }
  if (value.includes("rally")) {
    return "gemma4";
  }
  return null;
}

export function inferCustomModelDtype(modelId, family) {
  const value = String(modelId || "").toLowerCase();
  if (family === "qwen3_5") {
    if (value.includes("q8-onnx")) {
      return QWEN35_WEBGPU_Q8_DTYPE;
    }
    if (value.endsWith("-text") || value.includes("-text/") || value.includes("-text")) {
      return QWEN35_WEBGPU_TEXT_DTYPE;
    }
    if (value.includes("q4vision")) {
      return QWEN35_WEBGPU_Q4_VISION_DTYPE;
    }
    if (value.includes("q4-onnx")) {
      return QWEN35_WEBGPU_DTYPE;
    }
  }
  if (family === "gemma4") {
    if (value.endsWith("-text") || value.includes("-text/") || value.includes("-text")) {
      return GEMMA4_WEBGPU_TEXT_DTYPE;
    }
    if (value.includes("q4f16") || value.includes("rally-2b")) {
      return GEMMA4_WEBGPU_DTYPE;
    }
  }
  return null;
}

export function formatPresetSummary(preset) {
  if (!preset) {
    return "Custom model ID. Use a public Alkahest ONNX repo with a Transformers.js-compatible package layout.";
  }
  const dtypeLabel = typeof preset.dtype === "string" ? preset.dtype : JSON.stringify(preset.dtype);
  return `${preset.label} | ${preset.modalities} | ${preset.approxDownload} first load | ${dtypeLabel} | ${preset.note}`;
}

export async function clearBrowserModelCache() {
  if (!globalThis.caches) {
    return {
      supported: false,
      deleted: [],
    };
  }

  const cacheKey = env.cacheKey || "transformers-cache";
  const keys = await globalThis.caches.keys();
  const deleted = [];
  for (const key of keys) {
    if (key === cacheKey || key.startsWith(`${cacheKey}-`) || key.includes(cacheKey)) {
      if (await globalThis.caches.delete(key)) {
        deleted.push(key);
      }
    }
  }

  return {
    supported: true,
    deleted,
  };
}

export function createBrowserChatRuntime({
  defaultModelId = DEFAULT_MODEL_ID,
  defaultDevice = "webgpu",
  defaultDtype = "q4f16",
} = {}) {
  let activeModelId = null;
  let configPromise = null;
  let processorPromise = null;
  let modelPromise = null;
  let activeFamily = null;
  let activeDtype = null;
  let activeTextOnly = null;
  let activeAuthToken = null;

  function reset(modelId, family, dtype, textOnly, authToken) {
    const sameDtype = JSON.stringify(dtype) === JSON.stringify(activeDtype);
    if (
      modelId !== activeModelId ||
      family !== activeFamily ||
      !sameDtype ||
      textOnly !== activeTextOnly ||
      authToken !== activeAuthToken
    ) {
      activeModelId = modelId;
      activeFamily = family;
      activeDtype = dtype;
      activeTextOnly = textOnly;
      activeAuthToken = authToken;
      configPromise = null;
      processorPromise = null;
      modelPromise = null;
    }
  }

  async function load({
    modelId = defaultModelId,
    modelFamily,
    dtype,
    device = defaultDevice,
    textOnly = true,
    authToken = "",
    onProgress,
  } = {}) {
    if (!globalThis.navigator?.gpu && device === "webgpu") {
      throw new Error("WebGPU is not available in this browser.");
    }

    const preset = findModelPreset(modelId);
    const family = modelFamily ?? preset?.family ?? inferModelFamily(modelId) ?? "gemma4";
    const resolvedDtype = dtype ?? preset?.dtype ?? inferCustomModelDtype(modelId, family) ?? defaultDtype;
    const ProcessorClass = resolveProcessorClass(family);
    const ModelClass = resolveModelClass(family, { textOnly });
    const readyMessage = textOnly
      ? "Text chat ready. Multimodal sessions will load on first media prompt."
      : "Multimodal sessions ready.";
    const sessionsWarm =
      modelId === activeModelId &&
      family === activeFamily &&
      JSON.stringify(resolvedDtype) === JSON.stringify(activeDtype) &&
      textOnly === activeTextOnly &&
      processorPromise &&
      modelPromise;

    const hubOptions = authToken ? { token: authToken } : {};
    configureHubAuth(authToken);

    reset(modelId, family, resolvedDtype, textOnly, authToken);
    if (sessionsWarm) {
      onProgress?.({ status: "sessions_warm", textOnly }, "Runtime sessions already warm.");
    } else {
      onProgress?.(
        { status: textOnly ? "loading_text" : "loading_multimodal" },
        textOnly ? "Loading text sessions..." : "Loading multimodal sessions...",
      );
    }

    processorPromise ||= (async () => {
      onProgress?.({ status: "loading_processor" }, "Loading processor...");
      const processor = await ProcessorClass.from_pretrained(modelId, hubOptions);
      onProgress?.({ status: "processor_ready" }, "Processor ready.");
      return processor;
    })();
    configPromise ||= (async () => {
      onProgress?.({ status: "loading_config" }, "Loading config...");
      const config = await AutoConfig.from_pretrained(modelId, hubOptions);
      onProgress?.({ status: "config_ready" }, "Config ready.");
      return sanitizeBrowserConfig(config, resolvedDtype);
    })();
    modelPromise ||= (async () => {
      const config = await configPromise;
      onProgress?.({ status: "loading_model_sessions" }, "Loading ONNX sessions...");
      return ModelClass.from_pretrained(modelId, {
        ...hubOptions,
        config,
        device,
        dtype: resolvedDtype,
        progress_callback: (info) => {
          onProgress?.(info, makeProgressMessage(info));
        },
      });
    })();

    const [processor, model] = await Promise.all([processorPromise, modelPromise]);
    onProgress?.({ status: "done", textOnly }, readyMessage);
    return {
      processor,
      model,
      modelId: activeModelId,
      family: activeFamily,
      dtype: activeDtype,
      textOnly: activeTextOnly,
      readyMessage,
      preset,
    };
  }

  async function generate({
    modelId = activeModelId ?? defaultModelId,
    modelFamily,
    dtype,
    messages,
    systemPrompt = "",
    maxNewTokens = 160,
    temperature = 0.85,
    topP = 0.92,
    topK = 64,
    repetitionPenalty = 1.05,
    authToken = "",
    onToken,
    onProgress,
  }) {
    const { promptMessages, imageSources, audioSources, videoSources } = normalizeMessages([
      ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
      ...(messages ?? []),
    ]);
    const textOnly = imageSources.length === 0 && audioSources.length === 0 && videoSources.length === 0;

    const { family, processor, model } = await load({
      modelId,
      modelFamily,
      dtype,
      authToken,
      textOnly,
      onProgress,
    });
    const promptOptions = {
      add_generation_prompt: true,
    };
    if (family === "gemma4") {
      promptOptions.enable_thinking = false;
    } else if (family === "qwen3_5") {
      promptOptions.tokenizer_kwargs = { enable_thinking: false };
    }
    onProgress?.({ status: "formatting_prompt" }, "Formatting prompt...");
    const prompt = processor.apply_chat_template(promptMessages, promptOptions);

    onProgress?.({ status: "loading_media" }, "Preparing inputs...");
    const images =
      imageSources.length > 0 ? await Promise.all(imageSources.map((source) => load_image(source))) : [];
    const audios =
      audioSources.length > 0 ? await Promise.all(audioSources.map((source) => read_audio(source))) : [];
    const videos = videoSources.length > 0 ? [...videoSources] : [];

    const imageArg =
      images.length === 0 ? undefined : family === "qwen3_5" ? images : images.length === 1 ? images[0] : images;
    const audioArg = audios.length <= 1 ? audios[0] : audios;
    const videoArg =
      videos.length === 0 ? undefined : family === "qwen3_5" ? videos : videos.length === 1 ? videos[0] : videos;

    const inputs = await buildProcessorInputs(processor, family, prompt, imageArg, audioArg, videoArg);

    let streamedText = "";
    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk) => {
        streamedText += chunk;
        onToken?.(streamedText, chunk);
      },
    });

    onProgress?.({ status: "generating" }, "Generating...");
    const outputs = await model.generate({
      ...inputs,
      max_new_tokens: maxNewTokens,
      do_sample: temperature > 0,
      temperature,
      top_p: topP,
      top_k: topK,
      repetition_penalty: repetitionPenalty,
      streamer,
      ...(family === "qwen3_5" ? { return_dict_in_generate: true } : {}),
    });

    return decodeGeneratedText(processor, outputs?.sequences ?? outputs, inputs, streamedText);
  }

  async function dispose() {
    if (!modelPromise) {
      return;
    }
    const model = await modelPromise;
    if (typeof model?.dispose === "function") {
      await model.dispose();
    }
    processorPromise = null;
    modelPromise = null;
  }

  return {
    load,
    generate,
    dispose,
    get textOnly() {
      return activeTextOnly;
    },
    get modelId() {
      return activeModelId ?? defaultModelId;
    },
  };
}
