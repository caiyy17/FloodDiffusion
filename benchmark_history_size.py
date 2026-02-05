"""
Benchmark script to test a single history_size value for streaming generation.
Each run tests one hs value with independent model initialization.
Usage: python benchmark_history_size.py --hs 30 --tokens 1500
"""

import os
import time
import json
import numpy as np
import torch
from lightning import seed_everything
from torch_ema import ExponentialMovingAverage

from utils.initialize import instantiate, load_config
from utils.motion_process import StreamJointRecovery263

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB using nvidia-smi"""
    if torch.cuda.is_available():
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip().split('\n')[0])
        except:
            return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def load_models():
    """Load VAE and model from config (same as generate_ldf.py)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    cfg = load_config(config_path="configs/ldf_generate.yaml")
    seed_everything(cfg.seed)

    # Load VAE
    vae = instantiate(
        target=cfg.test_vae.target,
        cfg=None,
        hfstyle=False,
        **cfg.test_vae.params,
    )
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
    else:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
    vae.to(device)
    vae.eval()

    # Load model
    model = instantiate(
        target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
    )
    checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if "ema_state" in checkpoint:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(model.parameters())
    model.to(device)
    model.eval()

    return vae, model


def run_benchmark(history_size: int, total_tokens: int = 1500):
    """Run benchmark for a single history_size"""
    device = "cuda"
    os.makedirs("tmp/benchmark_analyze", exist_ok=True)
    
    print("=" * 60)
    print(f"Benchmark: history_size = {history_size}, tokens = {total_tokens}")
    print("=" * 60)
    
    # Load models
    print("\nLoading models...")
    vae, model = load_models()
    
    # Force T5 to GPU
    with torch.no_grad():
        _ = model.encode_text_with_cache(["test"], device)
    
    baseline_mem = get_gpu_memory_mb()
    print(f"Baseline memory: {baseline_mem:.0f} MB")
    
    # Create prompts (short prompts)
    text_list = [
        "walk forward",
        "stop and stretch",
        "run fast",
        "squat down",
        "spin kick",
        "dance",
        "jumping jacks",
        "bend over",
    ]
    
    tokens_per_prompt = total_tokens // len(text_list)
    text_end = [tokens_per_prompt * (i + 1) for i in range(len(text_list))]
    text_end[-1] = total_tokens
    
    text_end_with_zero = [0] + text_end
    durations = [t - b for t, b in zip(text_end_with_zero[1:], text_end_with_zero[:-1])]
    
    # Warmup: generate 20 tokens with hs=10 to initialize everything
    print("Warmup with hs=10, 20 tokens...")
    vae.clear_cache()
    model.init_generated(10, batch_size=1)
    warmup_first_chunk = True
    with torch.no_grad():
        for _ in range(20):
            x = {"text": ["warmup test"]}
            output = model.stream_generate_step(x, first_chunk=warmup_first_chunk)
            output = output["generated"]
            _ = vae.stream_decode(output[0][None, :], first_chunk=warmup_first_chunk)[0]
            warmup_first_chunk = False
    vae.clear_cache()
    torch.cuda.empty_cache()
    print("Warmup done.\n")
    
    # Initialize for actual test
    vae.clear_cache()
    model.init_generated(history_size, batch_size=1)
    torch.cuda.synchronize()
    init_mem = get_gpu_memory_mb()
    print(f"Memory after init: {init_mem:.0f} MB\n")
    
    # Initialize stream joint recovery
    stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    
    # Run generation
    step_times = []
    stream_joints = []
    first_chunk = True
    total_step = 0
    
    try:
        with torch.no_grad():
            for text_item, duration in zip(text_list, durations):
                for i in range(duration):
                    # Time the complete step (no sync, no mem check)
                    step_start = time.perf_counter()
                    
                    x = {"text": [text_item]}
                    output = model.stream_generate_step(x, first_chunk=first_chunk)
                    output = output["generated"]
                    decoded_g = vae.stream_decode(output[0][None, :], first_chunk=first_chunk)[0]
                    
                    # Process frame
                    decoded_g_np = decoded_g.cpu().numpy()
                    if decoded_g_np.ndim == 1:
                        frame_joints = stream_recovery.process_frame(decoded_g_np)
                        stream_joints.append(frame_joints)
                    else:
                        for frame_idx in range(decoded_g_np.shape[0]):
                            frame_joints = stream_recovery.process_frame(decoded_g_np[frame_idx])
                            stream_joints.append(frame_joints)
                    
                    first_chunk = False
                    step_end = time.perf_counter()
                    
                    step_time = step_end - step_start
                    step_times.append(step_time)
                    total_step += 1
                    
                    # Progress
                    if total_step % 100 == 0:
                        avg_time = np.mean(step_times[-100:]) * 1000
                        print(f"Step {total_step}/{total_tokens}: avg={avg_time:.1f}ms")
        
        vae.clear_cache()
        
        # Save joints
        stream_joints = np.array(stream_joints)
        np.save(f"tmp/benchmark_analyze/hs{history_size}_joints.npy", stream_joints)
        
        # Calculate stats
        step_times_ms = [t * 1000 for t in step_times]
        result = {
            "history_size": history_size,
            "total_tokens": total_tokens,
            "oom_error": False,
            "step_times_ms": step_times_ms,
            "step_time_mean_ms": float(np.mean(step_times_ms)),
            "step_time_std_ms": float(np.std(step_times_ms)),
            "step_time_min_ms": float(np.min(step_times_ms)),
            "step_time_max_ms": float(np.max(step_times_ms)),
        }
        
        with open(f"tmp/benchmark_analyze/hs{history_size}.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results for history_size = {history_size}")
        print(f"{'='*60}")
        print(f"Step time: {result['step_time_mean_ms']:.2f} ± {result['step_time_std_ms']:.2f} ms")
        print(f"Saved: hs{history_size}.json, hs{history_size}_joints.npy")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM Error at step {total_step}: {e}")
        result = {
            "history_size": history_size,
            "total_tokens": total_tokens,
            "oom_error": True,
            "error_message": str(e),
            "oom_at_step": total_step,
            "step_times_ms": [t * 1000 for t in step_times],
        }
        with open(f"tmp/benchmark_analyze/hs{history_size}.json", "w") as f:
            json.dump(result, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hs", type=int, required=True, help="History size to test")
    parser.add_argument("--tokens", type=int, default=1500, help="Total tokens to generate")
    args = parser.parse_args()
    
    run_benchmark(args.hs, args.tokens)


if __name__ == "__main__":
    main()
