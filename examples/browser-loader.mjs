import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer,
} from "@huggingface/transformers";

// Replace this with your own exported ONNX repo once training/export is done.
const modelId = "your-org/heretic-gemma-4-e4b-it-onnx";

let processorPromise;
let modelPromise;

export async function loadHereticModel() {
  processorPromise ||= AutoProcessor.from_pretrained(modelId);
  modelPromise ||= Gemma4ForConditionalGeneration.from_pretrained(modelId, {
    device: "webgpu",
    dtype: "q4f16",
    progress_callback: (info) => {
      if (info.status === "progress_total") {
        console.log(`Loading ${info.file}: ${info.progress}%`);
      }
    },
  });

  const [processor, model] = await Promise.all([processorPromise, modelPromise]);
  return { processor, model };
}

export async function runPrompt(userText) {
  const { processor, model } = await loadHereticModel();
  const messages = [
    {
      role: "system",
      content:
        "You are Heretic, an immersive roleplay assistant. Stay in character and answer directly.",
    },
    {
      role: "user",
      content: userText,
    },
  ];

  const prompt = processor.apply_chat_template(messages, {
    enable_thinking: false,
    add_generation_prompt: true,
  });

  const inputs = await processor(prompt, {
    add_special_tokens: false,
  });

  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function: (text) => {
      console.log(text);
    },
  });

  const outputs = await model.generate({
    ...inputs,
    max_new_tokens: 256,
    do_sample: true,
    temperature: 0.9,
    top_p: 0.95,
    top_k: 64,
    streamer,
  });

  const decoded = processor.batch_decode(
    outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
    { skip_special_tokens: true },
  );

  return decoded[0];
}

/*
Expected repo shape for `modelId`:

- config.json
- generation_config.json
- tokenizer.json
- tokenizer_config.json
- processor_config.json
- preprocessor_config.json
- chat_template.jinja
- onnx/*.onnx
- onnx/*.onnx_data*

If you move from the ONNX Community layout to a custom layout, loading will break.
*/
