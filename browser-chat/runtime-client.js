import { clearBrowserModelCache, createBrowserChatRuntime } from "../examples/browser-loader.mjs?v=3";

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

function createWorkerRuntime({
  defaultModelId,
  defaultDevice,
  defaultDtype,
} = {}) {
  const worker = new Worker(new URL("./runtime-worker.js", import.meta.url), { type: "module" });
  const pending = new Map();
  let nextRequestId = 1;
  let activeModelId = null;
  let activeTextOnly = null;

  function rejectAll(error) {
    for (const { reject } of pending.values()) {
      reject(error);
    }
    pending.clear();
  }

  worker.addEventListener("message", (event) => {
    const { id, kind, result, error, info, message, fullText, chunk } = event.data ?? {};
    const entry = pending.get(id);
    if (!entry) {
      return;
    }

    if (kind === "progress") {
      entry.onProgress?.(info, message);
      return;
    }

    if (kind === "token") {
      entry.onToken?.(fullText, chunk);
      return;
    }

    pending.delete(id);
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
  });

  worker.addEventListener("error", (event) => {
    rejectAll(event.error instanceof Error ? event.error : new Error(event.message || "Worker error"));
  });

  worker.addEventListener("messageerror", () => {
    rejectAll(new Error("Browser runtime worker could not deserialize a message."));
  });

  function request(type, payload = {}, { onProgress, onToken } = {}) {
    const id = nextRequestId++;
    return new Promise((resolve, reject) => {
      pending.set(id, {
        type,
        resolve,
        reject,
        onProgress,
        onToken,
      });
      worker.postMessage({ id, kind: type, payload });
    });
  }

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
      worker.terminate();
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
