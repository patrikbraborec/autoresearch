"""
Apify Actor entry point for autoresearch.
Wraps prepare.py + train.py workflow with Apify SDK for input/output/status.
"""

import asyncio
import io
import os
import sys
import time
import math
import gc
from dataclasses import asdict

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import torch.nn.functional as F

from apify import Actor


async def main() -> None:
    async with Actor:
        actor_input = await Actor.get_input() or {}

        depth = actor_input.get("depth", 4)
        time_budget = actor_input.get("timeBudget", 300)
        num_shards = actor_input.get("numShards", 10)
        device_batch_size = actor_input.get("deviceBatchSize", 8)

        Actor.log.info(f"Configuration: depth={depth}, time_budget={time_budget}s, num_shards={num_shards}, device_batch_size={device_batch_size}")

        # ---- Step 1: Data preparation ----
        await Actor.set_status_message("Preparing data: downloading shards and tokenizer...")

        from prepare import (
            MAX_SEQ_LEN, EVAL_TOKENS, Tokenizer,
            make_dataloader, evaluate_bpb, download_data, train_tokenizer,
        )

        download_data(num_shards)
        train_tokenizer()

        # ---- Step 2: Build model ----
        await Actor.set_status_message("Building model...")

        # Import train.py components (model, optimizer classes)
        from train import (
            GPTConfig, GPT, MuonAdamW,
            ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN,
            TOTAL_BATCH_SIZE, EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR,
            SCALAR_LR, WEIGHT_DECAY, ADAM_BETAS,
            WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC,
        )

        t_start = time.time()
        torch.manual_seed(42)

        tokenizer = Tokenizer.from_directory()
        vocab_size = tokenizer.get_vocab_size()
        Actor.log.info(f"Vocab size: {vocab_size:,}")

        base_dim = depth * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        num_heads = model_dim // HEAD_DIM

        config = GPTConfig(
            sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=WINDOW_PATTERN,
        )
        Actor.log.info(f"Model config: {asdict(config)}")

        model = GPT(config)
        model.init_weights()

        param_counts = model.num_scaling_params()
        num_params = param_counts["total"]
        Actor.log.info(f"Parameters: {num_params:,}")

        tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
        assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
        grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

        optimizer = model.setup_optimizer(
            unembedding_lr=UNEMBEDDING_LR,
            embedding_lr=EMBEDDING_LR,
            scalar_lr=SCALAR_LR,
            adam_betas=ADAM_BETAS,
            matrix_lr=MATRIX_LR,
            weight_decay=WEIGHT_DECAY,
        )

        train_loader = make_dataloader(tokenizer, device_batch_size, MAX_SEQ_LEN, "train")
        x, y, epoch = next(train_loader)

        # ---- Schedules ----
        def get_lr_multiplier(progress):
            if progress < WARMUP_RATIO:
                return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
            elif progress < 1.0 - WARMDOWN_RATIO:
                return 1.0
            else:
                cooldown = (1.0 - progress) / WARMDOWN_RATIO
                return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

        def get_muon_momentum(step):
            frac = min(step / 300, 1)
            return (1 - frac) * 0.85 + frac * 0.95

        def get_weight_decay(progress):
            return WEIGHT_DECAY * (1 - progress)

        # ---- Step 3: Training loop ----
        await Actor.set_status_message("Training started...")

        t_start_training = time.time()
        smooth_train_loss = 0
        total_training_time = 0
        step = 0

        while True:
            t0 = time.time()
            for micro_step in range(grad_accum_steps):
                loss = model(x, y)
                train_loss = loss.detach()
                loss = loss / grad_accum_steps
                loss.backward()
                x, y, epoch = next(train_loader)

            progress = min(total_training_time / time_budget, 1.0)
            lrm = get_lr_multiplier(progress)
            muon_momentum = get_muon_momentum(step)
            muon_weight_decay = get_weight_decay(progress)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_weight_decay
            optimizer.step()
            model.zero_grad(set_to_none=True)

            train_loss_f = train_loss.item()

            if train_loss_f > 100:
                Actor.log.error("Training FAILED - loss explosion detected")
                await Actor.push_data({"status": "FAILED", "reason": "loss_explosion", "step": step})
                await Actor.set_status_message("FAILED: loss explosion")
                return

            t1 = time.time()
            dt = t1 - t0

            if step > 2:
                total_training_time += dt

            ema_beta = 0.9
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
            pct_done = 100 * progress
            tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
            remaining = max(0, time_budget - total_training_time)

            # Update status every 5 steps
            if step % 5 == 0:
                status_msg = f"Step {step} | {pct_done:.1f}% | loss: {debiased_smooth_loss:.4f} | {remaining:.0f}s left"
                await Actor.set_status_message(status_msg)
                Actor.log.info(status_msg)

            step += 1

            if step > 2 and total_training_time >= time_budget:
                break

        total_tokens = step * TOTAL_BATCH_SIZE

        # ---- Step 4: Evaluation ----
        await Actor.set_status_message("Evaluating model (computing val_bpb)...")
        Actor.log.info("Starting evaluation...")

        model.eval()
        val_bpb = evaluate_bpb(model, tokenizer, device_batch_size)

        # ---- Step 5: Save model to key-value store ----
        await Actor.set_status_message("Saving trained model...")
        Actor.log.info("Saving model weights and config to key-value store...")

        # Save model weights
        weights_buffer = io.BytesIO()
        torch.save(model.state_dict(), weights_buffer)
        weights_bytes = weights_buffer.getvalue()

        kvs = await Actor.open_key_value_store()
        await kvs.set_value("model_weights.pt", weights_bytes, content_type="application/octet-stream")

        # Save model config as JSON for easy reconstruction
        import json
        config_json = json.dumps(asdict(config))
        await kvs.set_value("model_config.json", config_json, content_type="application/json")

        Actor.log.info(f"Model saved: weights ({len(weights_bytes) / 1e6:.1f} MB) + config")

        t_end = time.time()

        # Build download URLs
        kvs_info = await kvs.get_info()
        kvs_id = kvs_info["id"] if kvs_info else "unknown"
        weights_url = f"https://api.apify.com/v2/key-value-stores/{kvs_id}/records/model_weights.pt"
        config_url = f"https://api.apify.com/v2/key-value-stores/{kvs_id}/records/model_config.json"

        result = {
            "status": "SUCCESS",
            "val_bpb": round(val_bpb, 6),
            "training_seconds": round(total_training_time, 1),
            "total_seconds": round(t_end - t_start, 1),
            "total_tokens_M": round(total_tokens / 1e6, 1),
            "num_steps": step,
            "num_params_M": round(num_params / 1e6, 1),
            "depth": depth,
            "model_weights_url": weights_url,
            "model_config_url": config_url,
        }

        Actor.log.info(f"Results: {result}")
        await Actor.push_data(result)

        final_msg = f"Done! val_bpb={val_bpb:.6f} | {step} steps | {num_params/1e6:.1f}M params"
        await Actor.set_status_message(final_msg)
        Actor.log.info(final_msg)


if __name__ == "__main__":
    asyncio.run(main())
