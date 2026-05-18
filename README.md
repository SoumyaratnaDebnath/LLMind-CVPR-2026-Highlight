# LLMind : Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models

The app lets you:

- choose one of the bundled sample images or upload your own
- enter one or more question/answer pairs
- run the LLMind pipeline
- compare `LLMind` sampling against `Uniform` sampling

The current implementation uses `Qwen/Qwen2.5-VL-3B-Instruct` for visual question answering.

## Quick Start

### CPU

```bash
docker compose up --build
```

### NVIDIA GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Open:

```text
http://localhost:7860
```

## What You Need

- Docker
- Internet access on the first run so the Hugging Face models can be downloaded

For GPU mode, you also need:

- Linux host
- NVIDIA GPU
- NVIDIA drivers installed on the host
- NVIDIA Container Toolkit installed on the host

## What Gets Persisted

- `./logs` on your machine stores run outputs
- `llmind_hf_cache` Docker volume stores downloaded model files

That means the first run is slower, but later runs reuse the downloaded weights.

## Project Files

- [Dockerfile](/home/soumyara004/soumyaratna/Shareables/LLMind-Docker/Dockerfile): container image definition
- [docker-compose.yml](/home/soumyara004/soumyaratna/Shareables/LLMind-Docker/docker-compose.yml): default CPU-safe Docker setup
- [docker-compose.gpu.yml](/home/soumyara004/soumyaratna/Shareables/LLMind-Docker/docker-compose.gpu.yml): NVIDIA GPU override
- [app.py](/home/soumyara004/soumyaratna/Shareables/LLMind-Docker/app.py): Gradio UI
- [run_llmind.py](/home/soumyara004/soumyaratna/Shareables/LLMind-Docker/run_llmind.py): Python entrypoint for the pipeline
- [run.sh](/home/soumyara004/soumyaratna/Shareables/LLMind-Docker/run.sh): sample CLI run
- [example_data](/home/soumyara004/soumyaratna/Shareables/LLMind-Docker/example_data): bundled sample inputs
- [logs](/home/soumyara004/soumyaratna/Shareables/LLMind-Docker/logs): sample or generated outputs

## Running The UI

Start the app:

```bash
docker compose up --build
```

Or with GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Then open `http://localhost:7860` in a browser.

Inside the UI:

1. Use a sample image or upload your own image.
2. Provide question/answer rows.
3. Choose pixel budget, iterations, and image size.
4. Click `Run LLMind`.
5. Review the `LLMind` and `Uniform` outputs plus the answer comparison table.

## Running The CLI

To run the included shell script:

```bash
docker compose run --rm llmind bash run.sh
```

To call the Python entrypoint directly:

```bash
docker compose run --rm llmind python run_llmind.py \
  --image_path "./example_data/images/Image 4.png" \
  --percentage 0.01 \
  --mobius_layers 1 \
  --epochs 20 \
  --lr 1e-3 \
  --device auto \
  --scorer lhuman \
  --z_dim 64 \
  --hidden 512 \
  --exp_name "Image 4" \
  --log_every 1 \
  --log_dir "./logs" \
  --json_path "./example_data/info.json" \
  --vlm_model "qwen" \
  --img_size 300 \
  --param_limits 0.2 0.2 0.4 0.5
```

## Notes On Performance

- CPU mode is useful for basic validation and packaging checks.
- The Qwen2.5-VL model is large enough that GPU execution is the practical choice for normal use.
- The first successful run will spend time downloading model weights.

## Common Commands

Rebuild from scratch:

```bash
docker compose build --no-cache
```

Stop the app:

```bash
docker compose down
```

Stop the GPU variant:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

Delete the Hugging Face cache volume:

```bash
docker volume rm llmind_hf_cache
```

## Troubleshooting

If `docker` is not found:

- install Docker first

If GPU mode does not see your GPU:

- verify that NVIDIA drivers are installed on the host
- verify that NVIDIA Container Toolkit is installed
- run the GPU compose command, not the CPU-only one

If the app starts but inference is very slow:

- that is expected on CPU for this model
- switch to the GPU compose path if available

If model download fails on first run:

- check host internet access
- retry after the network issue is fixed

## Docker Details

More detailed Docker-specific notes are in [DOCKER.md](https://github.com/SoumyaratnaDebnath/LLMind-CVPR-2026-Highlight/blob/main/DOCKER.md).
