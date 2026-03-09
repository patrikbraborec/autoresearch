# autoresearch — Apify Actor fork

This is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that wraps the autonomous LLM training setup as an [Apify Actor](https://apify.com/actors) — a serverless cloud program that can be run, scheduled, and integrated via API.

## What's different from the original

- **Runs on CPU** — removed CUDA/GPU dependencies (Flash Attention 3, `kernels` package, pinned memory, bfloat16 casts) and replaced them with CPU-compatible equivalents (PyTorch SDPA attention, standard tensors)
- **Packaged as an Apify Actor** — `main.py` wraps the prepare + train workflow with the Apify SDK, providing structured input, live status updates, and output via Apify's dataset and key-value store
- **Configurable via Actor input** — model depth, time budget, number of data shards, and batch size are all adjustable through the Actor's input schema
- **Model artifacts saved** — trained model weights and config are saved to Apify's key-value store with download URLs included in the output

## How it works

The Actor runs a single GPT pretraining experiment:

1. Downloads training data shards and trains a BPE tokenizer
2. Builds a GPT model with configurable depth
3. Trains for a fixed time budget (default 5 minutes)
4. Evaluates validation bits-per-byte (val_bpb — lower is better)
5. Saves model weights and config, outputs results

## Input parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | 4 | Number of transformer layers |
| `timeBudget` | 300 | Training time in seconds |
| `numShards` | 10 | Number of training data shards to download |
| `deviceBatchSize` | 8 | Batch size (reduce if running out of memory) |

## Local development

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Data prep (one-time, ~2 min)
uv run prepare.py

# Run a training experiment directly
uv run train.py

# Or run as an Actor locally
uv run python main.py
```

## Docker

```bash
docker build -t autoresearch .
docker run autoresearch
```

## Project structure

```
main.py         — Apify Actor entry point (wraps prepare + train)
prepare.py      — constants, data prep + runtime utilities
train.py        — model, optimizer, training loop (CPU-adapted)
.actor/         — Actor configuration and input schema
Dockerfile      — CPU-only Docker build
program.md      — agent instructions (for autonomous research mode)
```

## License

MIT
