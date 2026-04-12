import * as Transformers from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const { AutoModelForVision2Seq, AutoProcessor, TextStreamer, env, load_image, read_audio } =
  Transformers;

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

const QWEN_PUBLIC_DTYPE = {
  embed_tokens: "q4",
  vision_encoder: "fp16",
  decoder_model_merged: "q4",
};

export const DEFAULT_MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX";

export const DEFAULT_MODEL_PRESETS = [
  {
    label: "Qwen 3.5 0.8B ONNX",
    modelId: "onnx-community/Qwen3.5-0.8B-ONNX",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~0.9-1.1 GB",
    dtype: QWEN_PUBLIC_DTYPE,
    note: "Best default consumer browser tier in this lineup.",
  },
  {
    label: "Alkahest Sheena 0.8B",
    modelId: "alkahest-ai/sheena-0.8b",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~0.9-1.1 GB",
    dtype: "q4f16",
    note: "Use this after the direct Qwen 0.8B conversion is published.",
  },
  {
    label: "Alkahest Sheena 0.8B RP",
    modelId: "alkahest-ai/sheena-0.8b-rp",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~0.9-1.1 GB",
    dtype: "q4f16",
    note: "Use this after the tuned Qwen 0.8B roleplay package is published.",
  },
  {
    label: "Qwen 3.5 2B ONNX",
    modelId: "onnx-community/Qwen3.5-2B-ONNX",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~1.5-1.8 GB",
    dtype: QWEN_PUBLIC_DTYPE,
    note: "Stronger desktop-tier Qwen preset with a materially larger first load.",
  },
  {
    label: "Alkahest Sheena 2B",
    modelId: "alkahest-ai/sheena-2b",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~1.5-1.8 GB",
    dtype: "q4f16",
    note: "Use this after the direct Qwen 2B conversion is published.",
  },
  {
    label: "Alkahest Sheena 2B RP",
    modelId: "alkahest-ai/sheena-2b-rp",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~1.5-1.8 GB",
    dtype: "q4f16",
    note: "Use this after the tuned Qwen 2B roleplay package is published.",
  },
  {
    label: "Gemma 4 E2B ONNX",
    modelId: "onnx-community/gemma-4-E2B-it-ONNX",
    family: "gemma4",
    modalities: "text + image + audio",
    approxDownload: "~3.4-3.5 GB",
    dtype: "q4f16",
    note: "Public Gemma 4 E2B baseline. Heavy, but still the most realistic Gemma browser lane.",
  },
  {
    label: "Alkahest Rally 2B",
    modelId: "alkahest-ai/rally-2b",
    family: "gemma4",
    modalities: "text + image + audio",
    approxDownload: "~3.4-3.5 GB",
    dtype: "q4f16",
    note: "Use this after the direct Gemma 4 E2B conversion is published.",
  },
  {
    label: "Alkahest Rally 2B RP",
    modelId: "alkahest-ai/rally-2b-rp",
    family: "gemma4",
    modalities: "text + image + audio",
    approxDownload: "~3.4-3.5 GB",
    dtype: "q4f16",
    note: "Use this after the tuned Gemma 4 E2B roleplay package is published.",
  },
  {
    label: "Qwen 3.5 4B ONNX",
    modelId: "onnx-community/Qwen3.5-4B-ONNX",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~3.1 GB",
    dtype: QWEN_PUBLIC_DTYPE,
    note: "High-quality desktop Qwen preset. Too large for broad mobile/browser use.",
  },
  {
    label: "Alkahest Sheena 4B",
    modelId: "alkahest-ai/sheena-4b",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~3.1 GB",
    dtype: "q4f16",
    note: "Use this after the direct Qwen 4B conversion is published.",
  },
  {
    label: "Alkahest Sheena 4B RP",
    modelId: "alkahest-ai/sheena-4b-rp",
    family: "qwen3_5",
    modalities: "text + image",
    approxDownload: "~3.1 GB",
    dtype: "q4f16",
    note: "Use this after the tuned Qwen 4B roleplay package is published.",
  },
  {
    label: "Alkahest Rally 4B",
    modelId: "alkahest-ai/rally-4b",
    family: "gemma4",
    modalities: "text + image + audio",
    approxDownload: "~5.2 GB",
    dtype: "q4f16",
    note: "Desktop-only Gemma tier. Not a realistic broad consumer default.",
  },
  {
    label: "Alkahest Rally 4B RP",
    modelId: "alkahest-ai/rally-4b-rp",
    family: "gemma4",
    modalities: "text + image + audio",
    approxDownload: "~5.2 GB",
    dtype: "q4f16",
    note: "Use this after the tuned Gemma 4 E4B roleplay package is published.",
  },
];

function normalizeMessages(messages) {
  const promptMessages = [];
  const imageSources = [];
  const audioSources = [];

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
      }
    }

    if (content.length > 0) {
      promptMessages.push({
        role: message.role,
        content,
      });
    }
  }

  return { promptMessages, imageSources, audioSources };
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
    return "Model ready.";
  }
  if (typeof info.status === "string") {
    return info.file ? `${info.status}: ${info.file}` : info.status;
  }
  return "Loading model...";
}

function resolveModelClass(family) {
  const specific =
    family === "gemma4"
      ? Transformers.Gemma4ForConditionalGeneration
      : Transformers.Qwen3_5ForConditionalGeneration;
  if (typeof specific?.from_pretrained === "function") {
    return specific;
  }
  if (typeof AutoModelForVision2Seq?.from_pretrained === "function") {
    return AutoModelForVision2Seq;
  }
  throw new Error(`No compatible Transformers.js class is available for ${family}.`);
}

export function findModelPreset(modelId) {
  return DEFAULT_MODEL_PRESETS.find((preset) => preset.modelId === modelId) ?? null;
}

export function formatPresetSummary(preset) {
  if (!preset) {
    return "Custom model ID. Use a public ONNX repo with a Transformers.js-compatible package layout.";
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
  let processorPromise = null;
  let modelPromise = null;
  let activeFamily = null;
  let activeDtype = null;

  function reset(modelId, family, dtype) {
    const sameDtype = JSON.stringify(dtype) === JSON.stringify(activeDtype);
    if (modelId !== activeModelId || family !== activeFamily || !sameDtype) {
      activeModelId = modelId;
      activeFamily = family;
      activeDtype = dtype;
      processorPromise = null;
      modelPromise = null;
    }
  }

  async function load({
    modelId = defaultModelId,
    modelFamily,
    dtype,
    device = defaultDevice,
    onProgress,
  } = {}) {
    if (!globalThis.navigator?.gpu && device === "webgpu") {
      throw new Error("WebGPU is not available in this browser.");
    }

    const preset = findModelPreset(modelId);
    const family = modelFamily ?? preset?.family ?? "gemma4";
    const resolvedDtype = dtype ?? preset?.dtype ?? defaultDtype;
    const ModelClass = resolveModelClass(family);

    reset(modelId, family, resolvedDtype);
    onProgress?.({ status: "loading" }, "Loading model...");

    processorPromise ||= AutoProcessor.from_pretrained(modelId);
    modelPromise ||= ModelClass.from_pretrained(modelId, {
      device,
      dtype: resolvedDtype,
      progress_callback: (info) => {
        onProgress?.(info, makeProgressMessage(info));
      },
    });

    const [processor, model] = await Promise.all([processorPromise, modelPromise]);
    onProgress?.({ status: "done" }, "Model ready.");
    return {
      processor,
      model,
      modelId: activeModelId,
      family: activeFamily,
      dtype: activeDtype,
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
    const { promptMessages, imageSources, audioSources } = normalizeMessages([
      ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
      ...(messages ?? []),
    ]);

    const { family, processor, model } = await load({
      modelId,
      modelFamily,
      dtype,
      onProgress,
    });
    const promptOptions = {
      add_generation_prompt: true,
    };
    if (family === "gemma4") {
      promptOptions.enable_thinking = false;
    }
    const prompt = processor.apply_chat_template(promptMessages, promptOptions);

    const images =
      imageSources.length > 0 ? await Promise.all(imageSources.map((source) => load_image(source))) : [];
    const audios =
      audioSources.length > 0 ? await Promise.all(audioSources.map((source) => read_audio(source))) : [];

    const imageArg = images.length <= 1 ? images[0] : images;
    const audioArg = audios.length <= 1 ? audios[0] : audios;

    let inputs;
    if (family === "gemma4") {
      if (imageArg && audioArg) {
        inputs = await processor(prompt, imageArg, audioArg, { add_special_tokens: false });
      } else if (imageArg) {
        inputs = await processor(prompt, imageArg, { add_special_tokens: false });
      } else if (audioArg) {
        inputs = await processor(prompt, undefined, audioArg, { add_special_tokens: false });
      } else {
        inputs = await processor(prompt, { add_special_tokens: false });
      }
    } else if (imageArg) {
      inputs = await processor(prompt, imageArg, { add_special_tokens: false });
    } else {
      inputs = await processor(prompt, { add_special_tokens: false });
    }

    let streamedText = "";
    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk) => {
        streamedText += chunk;
        onToken?.(streamedText, chunk);
      },
    });

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
    get modelId() {
      return activeModelId ?? defaultModelId;
    },
  };
}
