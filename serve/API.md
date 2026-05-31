# BiltIQ Local Inference API (laptop, roaming)

OpenAI-compatible API served by vLLM on this laptop, with **TurboQuant 3-bit KV
cache compression** active. Intended for an application running on the **same
machine** (bound to localhost only).

## Endpoint

| | |
|------------|------------------------------------------------|
| Base URL   | `http://localhost:8202/v1`                     |
| Host bind  | `127.0.0.1` (localhost only — no LAN exposure)  |
| Model name | `qwen3.5-4b`                                    |
| Auth       | Bearer token (required)                         |
| API key    | stored at `~/.config/biltiq_vllm_api_key` (chmod 600) |

Read the key:

```bash
cat ~/.config/biltiq_vllm_api_key
```

> The key is **not** committed to git. To rotate it: write a new value to that
> file and restart the service (`systemctl --user restart biltiq-qwen`).

## Model & runtime

- **Qwen3.5-4B-AWQ-4bit** — W4A16 (4-bit weights, 16-bit activations), multimodal,
  hybrid linear-attention + full-attention, 4096-token context (`--max-model-len`).
- **TurboQuant 3-bit KV compression** on the 8 full-attention layers
  (`~5.12x` KV ratio; the linear-attention layers carry no KV cache). W4A16 base
  keeps numerics clean — primes/arithmetic come out correct.
- Greedy/eager, `--max-num-seqs 1` (single-stream, laptop-sized).

## Quick start (curl)

```bash
KEY=$(cat ~/.config/biltiq_vllm_api_key)
curl -s http://localhost:8202/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "List the first 6 prime numbers."}],
    "temperature": 0,
    "max_tokens": 80,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

- `chat_template_kwargs.enable_thinking`: `false` → crisp final answer; omit (or
  `true`) → the model emits its reasoning trace first.
- No/incorrect key → `401`.

## Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8202/v1",
    api_key=open("/home/atc/.config/biltiq_vllm_api_key").read().strip(),
)

resp = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "What is 21 + 21?"}],
    temperature=0,
    max_tokens=64,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

## Endpoints

Standard vLLM OpenAI surface, all under `http://localhost:8202`:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET  /v1/models`
- `GET  /health` (no auth)

## Managing the service (systemd --user)

```bash
systemctl --user status   biltiq-qwen     # state / recent logs
systemctl --user restart  biltiq-qwen     # restart (Qwen init ~4.5 min)
systemctl --user stop     biltiq-qwen
journalctl --user -u biltiq-qwen -f       # follow logs
```

- Auto-starts on **login**. For start **before login** (true headless boot), run
  once: `sudo loginctl enable-linger atc`.
- Stopping the unit also tears down the `VLLM::EngineCore` child (cgroup kill), so
  no stranded GPU memory.

## Verify KV compression is active

```bash
cat ~/logs/bilt_turboquant_stats.json
# -> {"decode_fused": N>0, "decode_fallback": 0, "shadow_ratio": ~5.12, "layers": 8}
```

## Notes / limits

- **Localhost only.** To expose on the LAN: relaunch with `HOST=0.0.0.0` (and
  treat the key as a real secret). Default is `127.0.0.1`.
- Single sequence (`max_num_seqs=1`); not sized for concurrent load.
- KV compression here proves correctness + ratio; on this discrete GPU it does not
  yet *free* VRAM (the real paged cache is kept for the fallback path).
- The E4B serving variant (CPU embedding offload + tool calling, port 8201) is in
  `serve/launch_e4b_serve.sh`; E4B is W4A4 and less reliable for exact numerics.
