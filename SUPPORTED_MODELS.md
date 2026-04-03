# Supported Gemma 4 Models

## Model variants

| Model ID | Parameters | Type | Modalities | HuggingFace |
|----------|-----------|------|------------|-------------|
| `google/gemma-4-31B-it` | 31B | Instruction-tuned | Text + Vision | [Link](https://huggingface.co/google/gemma-4-31B-it) |
| `google/gemma-4-31B-pt` | 31B | Pretrained | Text + Vision | [Link](https://huggingface.co/google/gemma-4-31B-pt) |
| `google/gemma-4-12B-it` | 12B | Instruction-tuned | Text + Vision | [Link](https://huggingface.co/google/gemma-4-12B-it) |
| `google/gemma-4-12B-pt` | 12B | Pretrained | Text + Vision | [Link](https://huggingface.co/google/gemma-4-12B-pt) |
| `google/gemma-4-4B-it` | 4B | Instruction-tuned | Text | [Link](https://huggingface.co/google/gemma-4-4B-it) |
| `google/gemma-4-4B-pt` | 4B | Pretrained | Text | [Link](https://huggingface.co/google/gemma-4-4B-pt) |
| `google/gemma-4-1B-it` | 1B | Instruction-tuned | Text | [Link](https://huggingface.co/google/gemma-4-1B-it) |

## Recommended launch configurations

### gemma-4-31B-it (31B parameters)

**DGX Spark / GB10 (128GB unified memory, aarch64)**

```bash
vllm serve /path/to/gemma-4-31B-it \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --enable-prefix-caching
```

- Expected throughput: ~3.8 tok/s decode
- TTFT at 8K context: ~4.2s
- `--enforce-eager` required (CUDA graphs not stable on ARM for this model)
- Lower GPU utilization (0.55) because GB10 unified memory is shared with CPU

**Single A100 80GB (x86_64)**

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --enable-prefix-caching
```

- Expected throughput: ~45 tok/s decode
- TTFT at 8K context: ~1.1s

**2x A100 80GB (tensor parallel)**

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --enable-prefix-caching
```

**Single H100 80GB**

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --enable-prefix-caching
```

---

### gemma-4-12B-it (12B parameters)

**Single A100 80GB / H100**

```bash
vllm serve google/gemma-4-12B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --enable-prefix-caching
```

- Expected throughput: ~85 tok/s on A100

**RTX 4090 24GB (AWQ quantized)**

```bash
vllm serve google/gemma-4-12B-it \
  --trust-remote-code \
  --quantization awq \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 8
```

**DGX Spark / GB10**

```bash
vllm serve /path/to/gemma-4-12B-it \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.70 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --enable-prefix-caching
```

---

### gemma-4-4B-it (4B parameters)

**Any GPU with 16GB+ VRAM**

```bash
vllm serve google/gemma-4-4B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 32
```

**RTX 3090 / 4090 24GB**

```bash
vllm serve google/gemma-4-4B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --enable-prefix-caching
```

---

### gemma-4-1B-it (1B parameters)

**Any GPU with 8GB+ VRAM**

```bash
vllm serve google/gemma-4-1B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --enable-prefix-caching
```

---

## Common flags explained

| Flag | Purpose |
|------|---------|
| `--trust-remote-code` | Required for Gemma 4 custom code in HuggingFace config |
| `--enforce-eager` | Disables CUDA graphs; required on ARM/GB10, optional on x86_64 |
| `--gpu-memory-utilization` | Fraction of GPU memory to use (lower for unified memory systems) |
| `--max-model-len` | Maximum sequence length (tokens). Reduce if OOM. |
| `--max-num-seqs` | Maximum concurrent sequences. Reduce for lower memory usage. |
| `--enable-prefix-caching` | Reuses KV cache for shared prefixes. Recommended for chat. |
| `--tensor-parallel-size N` | Split model across N GPUs |
| `--quantization awq` | Use AWQ quantization (requires AWQ-quantized model weights) |

## Multimodal usage (vision)

The 12B and 31B `-it` variants support image inputs via the OpenAI-compatible API:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="google/gemma-4-31B-it",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
    max_tokens=256,
)

print(response.choices[0].message.content)
```

## Tool calling / function calling

Gemma 4 supports native tool calling:

```python
response = client.chat.completions.create(
    model="google/gemma-4-31B-it",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
)
```

Requires launching vLLM with `--tool-call-parser gemma4`.
