# Docker Setup

This project now ships with a Docker image and Compose files.

## What It Does

- Runs the Gradio app on `http://localhost:7860`
- Persists Hugging Face model downloads in a named Docker volume
- Persists run outputs in the local `./logs` directory
- Keeps the sample `example_data` directory mounted read-only

## Files

- `Dockerfile`: base image and Python environment
- `docker-compose.yml`: standard CPU-safe setup
- `docker-compose.gpu.yml`: optional GPU override for NVIDIA hosts

## Important Note

The app uses `Qwen/Qwen2.5-VL-3B-Instruct`. That model is large enough that CPU inference is mainly useful for validation, not for comfortable day-to-day use. For normal usage, prefer the GPU compose override.

## CPU Run

```bash
docker compose up --build
```

Then open:

```text
http://localhost:7860
```

## GPU Run

Requirements:

- Linux host
- NVIDIA GPU
- NVIDIA Container Toolkit installed on the host

Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Then open:

```text
http://localhost:7860
```

## Run The CLI Instead Of Gradio

The default container command starts `app.py`. To run the existing CLI flow instead:

```bash
docker compose run --rm llmind bash run.sh
```

Or call the Python entrypoint directly:

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

## Model Cache

Hugging Face downloads are stored in the named volume:

```text
llmind_hf_cache
```

Remove it if you want to force a clean model redownload:

```bash
docker volume rm llmind_hf_cache
```

## Notes

- First inference will take longer because model weights are downloaded.
- The Docker image installs PyTorch separately so the CPU build and the GPU build can use different wheel indexes without changing the project code.
