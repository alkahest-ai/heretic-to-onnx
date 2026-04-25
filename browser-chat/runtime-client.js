import { clearBrowserModelCache, createBrowserChatRuntime } from "../examples/browser-loader.mjs?v=8";

const DOWNLOAD_STALL_MS = 90_000;
const PHASE_STALL_MS = 120_000;

function errorFromPayload(error) {
  const normalized = new Error(error?.message || "Unknown browser runtime error");
  if (error?.stack) {
    normalized.stack = error.stack;
  }
  return normalized;
}

function serializePayload(payload) {
  if (!payload || typeof payload !== "object") {
    return payload;
  }

  return Object.fromEntries(
    Object.entries(payload).filter(([, value]) => typeof value !== "function" && value !== undefined),
  );
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

function progressSignature(info) {
  if (!info || typeof info !== "object") {
    return "no-progress";
  }
  return [
    info.status ?? "",
    info.file ?? "",
    Number.isFinite(info.loaded) ? Math.floor(info.loaded) : "",
    Number.isFinite(info.total) ? Math.floor(info.total) : "",
    Number.isFinite(info.progress) ? Math.floor(info.progress * 10) / 10 : "",
  ].join("|");
}

function isTransferProgress(info) {
  return Boolean(info?.file && Number.isFinite(info.loaded) && Number.isFinite(info.total) && info.total > 0);
}

function makeStallError(entry) {
  const info = entry.lastProgressInfo;
  const elapsed = Date.now() - entry.lastProgressAt;
  if (isTransferProgress(info)) {
    const loaded = formatByteCount(info.loaded);
    const total = formatByteCount(info.total);
    const percent = Number.isFinite(info.progress)
      ? Math.round(info.progress)
      : Math.round((info.loaded / info.total) * 100);
    return new Error(
      `Download stalled while fetching ${info.file} at ${percent}%${
        loaded && total ? ` (${loaded} / ${total})` : ""
      } for ${Math.round(elapsed / 1000)}s. The runtime worker was reset; clear cache and retry if this repeats.`,
    );
  }
  const status =
    info?.status === "done"
      ? "WebGPU session initialization"
      : info?.status
        ? String(info.status).replaceAll("_", " ")
        : entry.type;
  return new Error(
    `Runtime stalled during ${status} for ${Math.round(elapsed / 1000)}s. The runtime worker was reset; retry the load.`,
  );
}

function createWorkerRuntime({
  defaultModelId,
  defaultDevice,
  defaultDtype,
} = {}) {
  let worker = null;
  const pending = new Map();
  let nextRequestId = 1;
  let activeModelId = null;
  let activeTextOnly = null;

  function rejectAll(error) {
    for (const entry of pending.values()) {
      clearEntryTimer(entry);
      entry.reject(error);
    }
    pending.clear();
  }

  function clearEntryTimer(entry) {
    if (entry?.watchdogTimer) {
      clearInterval(entry.watchdogTimer);
      entry.watchdogTimer = null;
    }
  }

  function rejectEntry(id, error) {
    const entry = pending.get(id);
    if (!entry) {
      return;
    }
    clearEntryTimer(entry);
    pending.delete(id);
    entry.reject(error);
  }

  function resetWorker(error) {
    const oldWorker = worker;
    worker = null;
    activeModelId = null;
    activeTextOnly = null;
    if (oldWorker) {
      oldWorker.terminate();
    }
    rejectAll(error);
    ensureWorker();
  }

  function handleMessage(event) {
    const { id, kind, result, error, info, message, fullText, chunk } = event.data ?? {};
    const entry = pending.get(id);
    if (!entry) {
      return;
    }

    if (kind === "progress") {
      const signature = progressSignature(info);
      if (signature !== entry.lastProgressSignature) {
        entry.lastProgressSignature = signature;
        entry.lastProgressAt = Date.now();
        entry.lastProgressInfo = info;
      }
      entry.onProgress?.(info, message);
      return;
    }

    if (kind === "token") {
      entry.onToken?.(fullText, chunk);
      return;
    }

    pending.delete(id);
    clearEntryTimer(entry);
    if (kind === "error") {
      entry.reject(errorFromPayload(error));
      return;
    }

    if (entry.type === "load") {
      activeModelId = result?.modelId ?? activeModelId;
      activeTextOnly = result?.textOnly ?? activeTextOnly;
    } else if (entry.type === "dispose" || entry.type === "clearCache") {
      activeModelId = null;
      activeTextOnly = null;
    }

    entry.resolve(result);
  }

  function handleError(event) {
    rejectAll(event.error instanceof Error ? event.error : new Error(event.message || "Worker error"));
  }

  function handleMessageError() {
    rejectAll(new Error("Browser runtime worker could not deserialize a message."));
  }

  function ensureWorker() {
    if (worker) {
      return worker;
    }
    worker = new Worker(new URL("./runtime-worker.js?v=8", import.meta.url), { type: "module" });
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.addEventListener("messageerror", handleMessageError);
    return worker;
  }

  function request(type, payload = {}, { onProgress, onToken } = {}) {
    const id = nextRequestId++;
    return new Promise((resolve, reject) => {
      const entry = {
        type,
        resolve,
        reject,
        onProgress,
        onToken,
        lastProgressAt: Date.now(),
        lastProgressInfo: null,
        lastProgressSignature: "",
        watchdogTimer: null,
      };
      entry.watchdogTimer = setInterval(() => {
        const elapsed = Date.now() - entry.lastProgressAt;
        const limit = isTransferProgress(entry.lastProgressInfo) ? DOWNLOAD_STALL_MS : PHASE_STALL_MS;
        if (elapsed < limit) {
          return;
        }
        const error = makeStallError(entry);
        rejectEntry(id, error);
        resetWorker(error);
      }, 5_000);
      pending.set(id, entry);
      ensureWorker().postMessage({ id, kind: type, payload });
    });
  }

  ensureWorker();

  return {
    async supportsWorkerWebGpu() {
      const result = await request("capabilities");
      return Boolean(result?.workerWebGpu);
    },
    async load(options = {}) {
      return request("load", serializePayload(options), {
        onProgress: options.onProgress,
      });
    },
    async generate(options = {}) {
      return request("generate", serializePayload(options), {
        onProgress: options.onProgress,
        onToken: options.onToken,
      });
    },
    async dispose() {
      return request("dispose");
    },
    async clearCache() {
      return request("clearCache");
    },
    terminate() {
      rejectAll(new Error("Browser runtime worker terminated."));
      worker?.terminate();
      worker = null;
      activeModelId = null;
      activeTextOnly = null;
    },
    get modelId() {
      return activeModelId ?? defaultModelId;
    },
    get textOnly() {
      return activeTextOnly;
    },
  };
}

export function createBrowserChatRuntimeClient({
  defaultModelId,
  defaultDevice = "webgpu",
  defaultDtype = "q4f16",
} = {}) {
  let runtimeMode = "worker";
  let activeRuntime = null;
  let workerProbe = null;

  function ensureMainThreadRuntime() {
    runtimeMode = "main";
    activeRuntime = createBrowserChatRuntime({
      defaultModelId,
      defaultDevice,
      defaultDtype,
    });
    return activeRuntime;
  }

  function ensureWorkerRuntime() {
    if (!activeRuntime) {
      activeRuntime = createWorkerRuntime({
        defaultModelId,
        defaultDevice,
        defaultDtype,
      });
    }
    return activeRuntime;
  }

  async function resolveRuntime() {
    if (!globalThis.Worker) {
      return ensureMainThreadRuntime();
    }

    if (runtimeMode === "main") {
      return ensureMainThreadRuntime();
    }

    const runtime = ensureWorkerRuntime();
    if (!workerProbe) {
      workerProbe = runtime.supportsWorkerWebGpu().catch((error) => {
        runtime.terminate();
        activeRuntime = null;
        workerProbe = null;
        throw error;
      });
    }

    try {
      const workerWebGpu = await workerProbe;
      if (!workerWebGpu && defaultDevice === "webgpu") {
        runtime.terminate();
        activeRuntime = null;
        workerProbe = null;
        return ensureMainThreadRuntime();
      }
      return runtime;
    } catch {
      return ensureMainThreadRuntime();
    }
  }

  return {
    async load(options = {}) {
      const runtime = await resolveRuntime();
      return runtime.load(options);
    },
    async generate(options = {}) {
      const runtime = await resolveRuntime();
      return runtime.generate(options);
    },
    async dispose() {
      const runtime = await resolveRuntime();
      return runtime.dispose();
    },
    async clearCache() {
      if (runtimeMode === "worker") {
        const runtime = await resolveRuntime();
        if (runtimeMode === "worker") {
          return runtime.clearCache();
        }
      }

      const runtime = ensureMainThreadRuntime();
      await runtime.dispose();
      return clearBrowserModelCache();
    },
    get modelId() {
      return activeRuntime?.modelId ?? defaultModelId;
    },
    get textOnly() {
      return activeRuntime?.textOnly ?? null;
    },
    get mode() {
      return runtimeMode;
    },
  };
}
