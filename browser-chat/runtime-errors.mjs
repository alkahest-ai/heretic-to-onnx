export function formatRuntimeError(error) {
  const message = error?.message || String(error);
  if (message.includes("unsupported data type: 16")) {
    return `${message} This browser package still carries bfloat16 metadata and needs to be republished with the current packaging fix.`;
  }
  if (
    message.includes("Type parameter (T) of Optype (Add)") &&
    message.includes("tensor(float)") &&
    message.includes("tensor(float16)")
  ) {
    return `${message} This published q4f16 ONNX package was quantized with the older mixed-type conversion path and needs to be regenerated and republished.`;
  }
  return message;
}
