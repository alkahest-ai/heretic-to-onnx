import * as Transformers from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.1";

const { AutoConfig, AutoModelForVision2Seq, AutoProcessor, TextStreamer, env, load_image, read_audio } =
  Transformers;

env.allowLocalModels = true;
env.allowRemoteModels = true;
env.useBrowserCache = true;

export const PUBLIC_MODEL_OWNER = "thomasjvu";

function ownedModel(modelName) {
  return `${PUBLIC_MODEL_OWNER}/${modelName}`;
}

export const DEFAULT_MODEL_ID = ownedModel("rally-2b-rp");

export const DEFAULT_MODEL_PRESETS = [
  {
    label: "Alkahest 0.8B",
    modelId: ownedModel("alkahest-0.8b"),
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~1.2 GB",
    dtype: "q4f16",
    note: "Direct Alkahest browser package.",
  },
  {
    label: "Alkahest 0.8B V2",
    modelId: ownedModel("alkahest-0.8b-v2"),
    family: "qwen3_5",
    modalities: "text + image + video",
    approxDownload: "~1.9 GB",
    dtype: "q4f16",
    note: "Enhanced Alkahest browser package with video understanding.",
  },
  {
    label: "Alkahest 2B",
    modelId: ownedModel("alkahest-2b"),
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~3.3-3.4 GB",
    dtype: "q4f16",
    note: "Direct Alkahest browser package.",
  },
  {
    label: "Alkahest 2B V2",
    modelId: ownedModel("alkahest-2b-v2"),
    family: "qwen3_5",
    modalities: "text + image + video",
    approxDownload: "~3.3-3.4 GB",
    dtype: "q4f16",
    note: "Enhanced Alkahest browser package with video understanding.",
  },
  {
    label: "Alkahest 4B",
    modelId: ownedModel("alkahest-4b"),
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~5.1-5.2 GB",
    dtype: "q4f16",
    note: "Desktop-only Alkahest tier.",
  },
  {
    label: "Alkahest 4B V2",
    modelId: ownedModel("alkahest-4b-v2"),
    family: "qwen3_5",
    modalities: "text + image + video",
    approxDownload: "~5.1-5.2 GB",
    dtype: "q4f16",
    note: "Desktop-only enhanced Alkahest tier with video understanding.",
  },
  {
    label: "Rally 2B",
    modelId: ownedModel("rally-2b"),
    family: "gemma4",
    modalities: "text + image",
    approxDownload: "~3.4-3.5 GB",
    dtype: "q4f16",
    note: "Direct Rally browser package.",
  },
  {
    label: "Rally 2B V2",
    modelId: ownedModel("rally-2b-v2"),
    family: "gemma4",
    modalities: "text + image + audio + video",
    approxDownload: "~3.4-3.5 GB",
    dtype: "q4f16",
    note: "Enhanced Rally browser package with audio and video understanding.",
  },
  {
    label: "Rally 2B RP",
    modelId: ownedModel("rally-2b-rp"),
    family: "gemma4",
    modalities: "text + image + audio + video",
    approxDownload: "~3.4-3.5 GB",
    dtype: "q4f16",
    note: "Roleplay-tuned Rally browser package on the Gemma v2 multimodal contract.",
  },
  {
    label: "Rally 4B",
    modelId: ownedModel("rally-4b"),
    family: "gemma4",
    modalities: "text + image",
    approxDownload: "~5.2 GB",
    dtype: "q4f16",
    note: "Desktop-only Rally tier.",
  },
  {
    label: "Rally 4B V2",
    modelId: ownedModel("rally-4b-v2"),
    family: "gemma4",
    modalities: "text + image + audio + video",
    approxDownload: "~5.2 GB",
    dtype: "q4f16",
    note: "Desktop-only enhanced Rally tier with audio and video understanding.",
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

function sanitizeBrowserConfig(config, dtype) {
  if (!config || typeof config !== "object") {
    return config;
  }
  if (!["q4f16", "fp16"].includes(dtype)) {
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
        ? [Transformers.Qwen3_5ForCausalLM, Transformers.Qwen3ForCausalLM]
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
  const attempt = async (factory) => {
    try {
      return await factory();
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

  if (!imageArg && !audioArg && !videoArg) {
    attempts.push(() => processor(prompt, options));
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

export function formatPresetSummary(preset) {
  if (!preset) {
    return "Custom model ID. Use a public Alkahest or Rally ONNX repo with a Transformers.js-compatible package layout.";
  }
  return `${preset.label} | ${preset.modalities} | ${preset.approxDownload} first load | ${preset.note}`;
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

  function reset(modelId, family, dtype, textOnly) {
    const sameDtype = JSON.stringify(dtype) === JSON.stringify(activeDtype);
    if (modelId !== activeModelId || family !== activeFamily || !sameDtype || textOnly !== activeTextOnly) {
      activeModelId = modelId;
      activeFamily = family;
      activeDtype = dtype;
      activeTextOnly = textOnly;
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
    onProgress,
  } = {}) {
    if (!globalThis.navigator?.gpu && device === "webgpu") {
      throw new Error("WebGPU is not available in this browser.");
    }

    const preset = findModelPreset(modelId);
    const family = modelFamily ?? preset?.family ?? inferModelFamily(modelId) ?? "gemma4";
    const resolvedDtype = dtype ?? preset?.dtype ?? defaultDtype;
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

    reset(modelId, family, resolvedDtype, textOnly);
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
      const processor = await ProcessorClass.from_pretrained(modelId);
      onProgress?.({ status: "processor_ready" }, "Processor ready.");
      return processor;
    })();
    configPromise ||= (async () => {
      onProgress?.({ status: "loading_config" }, "Loading config...");
      const config = await AutoConfig.from_pretrained(modelId);
      onProgress?.({ status: "config_ready" }, "Config ready.");
      return sanitizeBrowserConfig(config, resolvedDtype);
    })();
    modelPromise ||= (async () => {
      const config = await configPromise;
      onProgress?.({ status: "loading_model_sessions" }, "Loading ONNX sessions...");
      return ModelClass.from_pretrained(modelId, {
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
      textOnly,
      onProgress,
    });
    const promptOptions = {
      add_generation_prompt: true,
    };
    if (family === "gemma4") {
      promptOptions.enable_thinking = false;
    }
    onProgress?.({ status: "formatting_prompt" }, "Formatting prompt...");
    const prompt = processor.apply_chat_template(promptMessages, promptOptions);

    onProgress?.({ status: "loading_media" }, "Preparing inputs...");
    const images =
      imageSources.length > 0 ? await Promise.all(imageSources.map((source) => load_image(source))) : [];
    const audios =
      audioSources.length > 0 ? await Promise.all(audioSources.map((source) => read_audio(source))) : [];
    const videos = videoSources.length > 0 ? [...videoSources] : [];

    const imageArg = images.length <= 1 ? images[0] : images;
    const audioArg = audios.length <= 1 ? audios[0] : audios;
    const videoArg = videos.length <= 1 ? videos[0] : videos;

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
    });

    const promptTokens = inputs.input_ids.dims.at(-1);
    const decoded = processor.batch_decode(outputs.slice(null, [promptTokens, null]), {
      skip_special_tokens: true,
    })[0];

    return (decoded ?? streamedText).trim();
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
