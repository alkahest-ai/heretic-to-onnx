import assert from "node:assert/strict";

import { formatRuntimeError } from "../browser-chat/runtime-errors.mjs";

const bf16Error = formatRuntimeError(new Error("unsupported data type: 16"));
assert.match(bf16Error, /needs to be republished/i);
assert.match(bf16Error, /bfloat16 metadata/i);

const mixedTypeError = formatRuntimeError(
  new Error(
    "Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: Type Error: Type parameter (T) of Optype (Add) bound to different types (tensor(float) and tensor(float16))",
  ),
);
assert.match(mixedTypeError, /needs to be regenerated and republished/i);
assert.match(mixedTypeError, /mixed-type conversion path/i);

const genericError = formatRuntimeError(new Error("plain failure"));
assert.equal(genericError, "plain failure");
