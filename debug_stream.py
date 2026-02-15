"""Debug script: print detailed streaming step info to tmp/debug_stream.txt"""
import os
import sys
import math
import torch
from lightning import seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.initialize import instantiate, load_config
from torch_ema import ExponentialMovingAverage
from utils.initialize import compare_statedict_and_parameters

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)

    # Load model only (no VAE needed for debug)
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

    # Config
    scheduler = model.time_scheduler
    chunk_size = scheduler.chunk_size
    steps = scheduler.steps
    seq_len = 30
    buf_len = seq_len * 2
    batch_size = 1

    text_list = ["walk in a circle.", "jump up."]
    text_end = [150, 250]
    text_end_with_zero = [0] + text_end
    durations = [t - b for t, b in zip(text_end_with_zero[1:], text_end_with_zero[:-1])]

    os.makedirs("tmp", exist_ok=True)
    f = open("tmp/debug_stream.txt", "w")

    def log(msg):
        f.write(msg + "\n")
        f.flush()

    log(f"=== Stream Debug ===")
    log(f"chunk_size={chunk_size}, steps={steps}, seq_len={seq_len}, buf_len={buf_len}")
    log(f"noise_type={scheduler.noise_type}")
    log(f"time_embedding_scale={model.time_embedding_scale}")
    log(f"cfg_config={model.cfg_config}")
    log(f"prediction_type={model.prediction_type}")
    log(f"param_dtype={model.param_dtype}")
    log(f"")

    # Init streaming
    model.init_generated(seq_len, batch_size=batch_size)

    condition_frame_global = 0
    commit_count = 0

    with torch.no_grad():
        for text_item, duration in zip(text_list, durations):
            for frame_i in range(duration):
                condition_frame_global += 1

                # Update conditions
                ik = model.input_keys
                x_input = {ik["text"]: [text_item]}
                x_input = model._extract_inputs(x_input)
                model.text_module.update_stream(x_input, device, model.param_dtype)
                model.condition_frames += 1

                # Rollback check
                if model.condition_frames > model.buf_len:
                    log(f"!!! ROLLBACK at condition_frames={model.condition_frames}")
                    model._rollback()

                committable_length, committable_steps = scheduler.get_committable(model.condition_frames)

                # Only log detail for first 20 condition frames
                verbose = condition_frame_global <= 20

                if verbose:
                    log(f"{'='*80}")
                    log(f"condition_frame={condition_frame_global}, text='{text_item}', "
                        f"condition_frames={model.condition_frames}, "
                        f"committable_length={committable_length}, committable_steps={committable_steps}, "
                        f"current_step={model.current_step}, current_commit={model.current_commit}")

                while model.current_step < committable_steps:
                    step = model.current_step
                    time_steps = scheduler.get_time_steps(
                        device, [model.buf_len] * batch_size, step)
                    time_schedules, time_schedules_derivative = scheduler.get_time_schedules(
                        device, [model.buf_len] * batch_size, time_steps)
                    noise_level, noise_level_derivative = scheduler.get_noise_levels(
                        device, [model.buf_len] * batch_size, time_schedules)
                    is_, ie_, os_, oe_ = scheduler.get_windows(
                        [model.buf_len] * batch_size, time_steps)

                    _, _, xt = scheduler.add_noise(
                        model.generated, noise_level, is_, ie_, os_, oe_, training=False)

                    noisy_input = [xt[i][:, -seq_len:, ...] for i in range(batch_size)]
                    ts = time_schedules[0][is_[0]:ie_[0]][-seq_len:]
                    cut_length = max(0, (ie_[0] - is_[0]) - seq_len)

                    os_idx, oe_idx = os_[0], oe_[0]
                    pred_os_idx = os_idx - is_[0] - cut_length
                    pred_oe_idx = oe_idx - is_[0] - cut_length

                    dt_vals = time_schedules_derivative[0][os_idx:oe_idx]
                    nl_vals = noise_level[0][os_idx:oe_idx]
                    ts_output = time_schedules[0][os_idx:oe_idx]

                    if verbose:
                        log(f"  step={step}, t={time_steps[0].item():.4f}")
                        log(f"    is={is_[0]}, ie={ie_[0]}, os={os_idx}, oe={oe_idx}, cut={cut_length}")
                        log(f"    pred_os={pred_os_idx}, pred_oe={pred_oe_idx}")
                        log(f"    noisy_input shape={noisy_input[0].shape}")
                        log(f"    ts_padded (first 15): {ts[:15].tolist()}")
                        log(f"    ts_at_output: {ts_output.tolist()}")
                        log(f"    noise_level_at_output: {nl_vals.tolist()}")
                        log(f"    dt_at_output: {dt_vals.tolist()}")
                        # Generated buffer stats at output positions
                        gen_out = model.generated[0][:, os_idx:oe_idx, ...]
                        log(f"    generated[os:oe] mean={gen_out.mean().item():.6f}, std={gen_out.std().item():.6f}")

                    # Actually run the model step
                    time_schedules_padded = torch.zeros(batch_size, seq_len, device=device)
                    for i in range(batch_size):
                        time_schedules_padded[i, :len(ts)] = ts

                    text_context = model.text_module.get_stream_context(
                        is_[0], ie_[0], seq_len)
                    null_context = model.text_module.get_null_context(
                        batch_size, device, model.param_dtype)
                    pred_text = model.model(
                        noisy_input,
                        time_schedules_padded * model.time_embedding_scale,
                        text_context, seq_len, y=None)
                    pred_null = model.model(
                        noisy_input,
                        time_schedules_padded * model.time_embedding_scale,
                        null_context, seq_len, y=None)
                    predicted_result = [
                        model.cfg_config["text_scale"] * pt + model.cfg_config["null_scale"] * pn
                        for pt, pn in zip(pred_text, pred_null)]

                    dt = time_schedules_derivative[0][os_idx:oe_idx][None, :, None, None]
                    nl = noise_level[0][os_idx:oe_idx][None, :, None, None]
                    nld = noise_level_derivative[0][os_idx:oe_idx][None, :, None, None]

                    for i in range(batch_size):
                        predicted_result_i = predicted_result[i]
                        if model.prediction_type == "vel":
                            predicted_vel = predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                        elif model.prediction_type == "x0":
                            predicted_vel = (
                                model.generated[i][:, os_idx:oe_idx, ...]
                                - predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                            ) / nl * nld
                        elif model.prediction_type == "noise":
                            predicted_vel = (
                                predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                                - model.generated[i][:, os_idx:oe_idx, ...]
                            ) / (1 - nl + dt) * nld

                        if verbose:
                            log(f"    pred_vel mean={predicted_vel.mean().item():.6f}, std={predicted_vel.std().item():.6f}")
                            update = predicted_vel * dt
                            log(f"    update mean={update.mean().item():.6f}, std={update.std().item():.6f}")

                        model.generated[i][:, os_idx:oe_idx, ...] += predicted_vel * dt

                    if verbose:
                        gen_out_after = model.generated[0][:, os_idx:oe_idx, ...]
                        log(f"    generated[os:oe] AFTER mean={gen_out_after.mean().item():.6f}, std={gen_out_after.std().item():.6f}")

                    model.current_step += 1

                # Commit
                if model.current_commit < committable_length:
                    output = [model.generated[i][:, model.current_commit:committable_length, ...]
                              for i in range(batch_size)]
                    output = model.postprocess(output)
                    output = [o * model.std + model.mean for o in output]
                    new_frames = committable_length - model.current_commit
                    if verbose or new_frames > 0:
                        log(f"  COMMIT: frames [{model.current_commit}:{committable_length}] = {new_frames} new frames")
                        log(f"    output shape={output[0].shape}, mean={output[0].mean().item():.4f}, std={output[0].std().item():.4f}")
                    model.current_commit = committable_length
                    commit_count += new_frames

                if condition_frame_global == 20:
                    log(f"\n... (truncated after 20 condition frames, total committed so far: {commit_count}) ...")
                    break
            if condition_frame_global >= 20:
                break

    f.close()
    print(f"Debug info written to tmp/debug_stream.txt ({commit_count} frames committed in first 20 steps)")


if __name__ == "__main__":
    main()
