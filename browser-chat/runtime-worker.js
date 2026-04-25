import { clearBrowserModelCache, createBrowserChatRuntime } from "../examples/browser-loader.mjs?v=5";

const runtime = createBrowserChatRuntime();

function serializeError(error) {
  return {
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack ?? "" : "",
  };
}

function postProgress(id, info, message) {
  self.postMessage({
    id,
    kind: "progress",
    info,
    message,
  });
}

function postToken(id, fullText, chunk) {
  self.postMessage({
    id,
    kind: "token",
    fullText,
    chunk,
  });
}

self.addEventListener("message", async (event) => {
  const { id, kind, payload = {} } = event.data ?? {};
  if (!id || !kind) {
    return;
  }

  try {
    switch (kind) {
      case "capabilities": {
        self.postMessage({
          id,
          kind: "result",
          result: {
            workerWebGpu: Boolean(self.navigator?.gpu),
          },
        });
        return;
      }

      case "load": {
        const result = await runtime.load({
          ...payload,
          onProgress: (info, message) => postProgress(id, info, message),
        });
        self.postMessage({
          id,
          kind: "result",
          result: {
            modelId: result.modelId,
            family: result.family,
            dtype: result.dtype,
            textOnly: result.textOnly,
            readyMessage: result.readyMessage,
            preset: result.preset,
          },
        });
        return;
      }

      case "generate": {
        const result = await runtime.generate({
          ...payload,
          onProgress: (info, message) => postProgress(id, info, message),
          onToken: (fullText, chunk) => postToken(id, fullText, chunk),
        });
        self.postMessage({
          id,
          kind: "result",
          result,
        });
        return;
      }

      case "dispose": {
        await runtime.dispose();
        self.postMessage({
          id,
          kind: "result",
          result: null,
        });
        return;
      }

      case "clearCache": {
        await runtime.dispose();
        const result = await clearBrowserModelCache();
        self.postMessage({
          id,
          kind: "result",
          result,
        });
        return;
      }

      default:
        throw new Error(`Unknown runtime request: ${kind}`);
    }
  } catch (error) {
    self.postMessage({
      id,
      kind: "error",
      error: serializeError(error),
    });
  }
});
