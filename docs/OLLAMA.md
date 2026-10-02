# Running converted models in Ollama (thinking mode included)

Any GGUF this app produces converts to an Ollama model with:

```bash
python tools/ollama_modelfile.py path/to/model.gguf --name my-model
ollama create my-model -f path/to/model.Modelfile
ollama run my-model --think          # thinking on
ollama run my-model                  # thinking off
```

## Why no TEMPLATE is needed (and why adding one breaks things)

Ollama does **not** execute the Jinja `tokenizer.chat_template` embedded in a
GGUF — its Go code has no Jinja engine (llama-server is launched with
`--no-jinja`). For Gemma 4, Ollama ships a **native Go renderer**
(`model/renderers/gemma4.go`) that renders chat turns, tool calls, and
thinking blocks, and `ollama create` auto-detects it from the GGUF's
`general.architecture = "gemma4"` metadata (verified against ollama/ollama
@ 6383a0f: `server/create.go` stamps `Renderer`/`Parser`/`stop` from the
architecture). The official `gemma4:26b` Modelfile's template is just the
13-byte passthrough `{{ .Prompt }}` — proof that all the formatting lives in
the renderer, not the template.

Consequences:

- A `FROM file.gguf` import of a converted Gemma 4 GGUF gets **native chat +
  thinking** automatically, same as `ollama pull gemma4`.
- Thinking is controlled by the request, not the prompt: `"think": true` in
  `/api/chat`, `--think` in the CLI. The renderer injects the `<|think|>`
  token; with `think: false` the `gemma4-large` variant emits an empty
  thought block (matches the official model's behavior).
- Do **not** add a `TEMPLATE` block to the Modelfile — on the renderer path
  it is dead code at best, and it overrides/bypasses the renderer at worst.
- The generated Modelfile pins `RENDERER`/`PARSER` explicitly so the file
  also works on Ollama builds whose arch auto-detect is missing or picks the
  small variant. If `ollama create` errors on the `RENDERER` line, your
  Ollama is old: update it (Gemma 4 support landed in 0.20.5+), or delete
  the `RENDERER`/`PARSER` lines and rely on auto-detect.

## Sampling parameters

The generated Modelfile ships the official Gemma 4 sampling config:
`temperature 1.0`, `top_k 64`, `top_p 0.95`, plus `stop "<turn|>"` and a
sane `num_ctx 32768` (raise it for long-context work; 26B Q4.3-class MoE
fits 32k comfortably in 16 GB).

## Multimodal / draft-model notes

- GGUFs converted here are **text-only** (the vision tower and audio encoder
  are not converted). Create exactly like above; do not reference a second
  `FROM` mmproj blob.
- The official `gemma4:26b` also ships an MTP draft model for speculative
  decoding. Our conversions don't include it; Ollama runs fine without
  (`draft_num_predict` is simply unused).
