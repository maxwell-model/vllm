<br>

<div align="center">
 <img src="assets/logo-en.png" alt="Maxwell-V1-Flash" width="400"/> <br>
</div>

<div align="center" style="line-height: 1;">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
&nbsp;&nbsp;
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Maxwell-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/maxwell-model/Maxwell-V1-Flash)

</div>

<div align="center">
    🖥️&nbsp;<a href="https://maxwell-llm.ai"><b>Official Website</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    💬&nbsp;<a href="https://github.com/maxwell-model/Maxwell-V1-Flash"><b>GitHub</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    📄 <a href=""><b>Technical Report（TODO 添加链接）</b></a>
</div>

---

## Deployment

The example has been validated on **NVIDIA H100 and B300** GPUs.

> [!TIP]
> The simplest way to get started is to launch from the prebuilt Docker image.
> See [examples/online_serving](https://github.com/maxwell-model/Maxwell-V1-Flash/tree/main/examples/online_serving).
> The steps below cover building from source.

### Prerequisites

- Linux with NVIDIA driver >= 580 (CUDA 13.0). On older drivers, install the
  `cuda-compat` package and set `VLLM_ENABLE_CUDA_COMPATIBILITY=1` and
  `VLLM_CUDA_COMPATIBILITY_PATH`; see
  [troubleshooting](docs/usage/troubleshooting.md).
- CUDA toolkit with `nvcc` (13.0 recommended), `git`,
  and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

### Install

```bash
git clone -b maxwell-v1-flash https://github.com/maxwell-model/vllm.git
cd vllm

uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install --no-deps -r requirements/mbsa.txt
```

### Serve on B300

A single GPU is enough:

```bash
vllm serve maxwell-model/Maxwell-V1-Flash \
  --trust-remote-code \
  --block-size 128 \
  --served-model-name Maxwell-V1-Flash \
  --port 8000 \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Serve on H100

Requires at least two GPUs. Use the same command with `--tensor-parallel-size 2`.

## Quickstart

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Maxwell-V1-Flash",
    "messages": [
      {
        "role": "user",
        "content": "Why is it harder to keep a room cool when the door is open on a hot day? Explain in 3-4 simple sentences."
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

## Maxwell-V1-Flash runtime options

### Environment variables

| Variable | Default | Purpose                                                                                  |
| --- | --- |------------------------------------------------------------------------------------------|
| `MAXWELL_MBSA_INDEX` | `fp8` | Index-cache format: `bf16`, `fp16`, `fp8` or `mxfp6` (`mxfp6` needs an SM100-family GPU) |
| `MAXWELL_MBSA_PARTIALS_FP8` | `1` | Store MBSA partial outputs in `fp8`; `0` selects `bf16`                                    |

## License and sources

vLLM and the Maxwell-V1-Flash integration are licensed under
[Apache License 2.0](LICENSE). Third-party components retain their original
licenses; see [NOTICE](NOTICE) for source attribution and modification summaries.
MBSA's component license texts and provenance records are retained in its
installed package. Model weights and tokenizer assets are subject to their
separate release terms.
