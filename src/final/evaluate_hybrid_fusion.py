#!/usr/bin/env python3
"""Validation-only fusion of Transformer and CHAD Episode Memory scores.

Required files in the same project:
  - evaluate_episode_memory_v2.py
  - chad_transformer_comparison.py
  - chad_windows_2s_4s_v2.npz
  - results_chad_gpu/model_transformer_seed{41,42,43}.pt

All fusion weights, score calibration, and thresholds are selected on the
validation split. Test labels are used only once for final reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from chad_transformer_comparison import Transformer
import evaluate_episode_memory_v2 as memory


def best_threshold(y, score):
    candidates = np.unique(np.quantile(score, np.linspace(.01, .99, 199)))
    values = [f1_score(y, score >= t, zero_division=0)
              for t in candidates]
    return float(candidates[int(np.argmax(values))])


def result_metrics(y, score, threshold):
    pred = score >= threshold
    return {
        "episode_roc_auc": float(roc_auc_score(y, score)),
        "episode_pr_auc": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive": int(y.sum()),
    }


class ValidationCDF:
    """Map arbitrary scores to validation-derived percentiles."""
    def __init__(self, validation_score):
        self.reference = np.sort(np.asarray(validation_score, dtype=float))

    def transform(self, score):
        score = np.asarray(score, dtype=float)
        rank = np.searchsorted(self.reference, score, side="right")
        return rank.astype(np.float64) / len(self.reference)


@torch.no_grad()
def transformer_scores(model, x, batch_size, device):
    model.eval()
    scores = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start:start + batch_size]).to(device)
        with torch.autocast(device_type=device.type,
                            enabled=device.type == "cuda"):
            logits = model(batch)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores)


def select_alpha(y, transformer_score, memory_score, step, pr_weight):
    best = None
    for alpha in np.arange(0.0, 1.0 + step / 2, step):
        score = alpha * transformer_score + (1.0 - alpha) * memory_score
        threshold = best_threshold(y, score)
        pr = average_precision_score(y, score)
        f1 = f1_score(y, score >= threshold, zero_division=0)
        objective = pr_weight * pr + (1.0 - pr_weight) * f1
        if best is None or objective > best[0] + 1e-12:
            best = objective, float(alpha), threshold, pr, f1
    return best


def prepare_memory(data, args):
    x = np.nan_to_num(data["x"].astype(np.float32),
                      nan=0, posinf=0, neginf=0)
    y = data["y"].astype(int)
    split = data["split"].astype(str)
    names = data["feature_names"].astype(str)
    sample_id, video, actor, pair, start, _ = memory.parsed_metadata(data)
    if np.all(pair == ""):
        raise ValueError("NPZ has no pair_id. Use chad_windows_2s_4s_v2.npz")

    actor_raw = memory.embedding(x, names, False)
    pair_raw = memory.embedding(x, names, True)
    train_normal = (split == "train") & (y == 0)
    ac, ass = memory.robust_fit(actor_raw[train_normal])
    pc, pss = memory.robust_fit(pair_raw[train_normal])
    actor_z = memory.robust_transform(actor_raw, ac, ass)
    pair_z = memory.robust_transform(pair_raw, pc, pss)

    global_model, global_ref = memory.fit_knn(actor_z[train_normal], args.k)
    pair_model, pair_global_ref = memory.fit_knn(pair_z[train_normal], args.k)
    actor_ref, pair_ref = memory.history_calibration(
        np.flatnonzero(train_normal), video, actor, pair, start,
        actor_z, pair_z, args.min_history, args.memory_size)
    calibrators = (memory.Calibrator(global_ref), memory.Calibrator(actor_ref),
                   memory.Calibrator(pair_global_ref),
                   memory.Calibrator(pair_ref))
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    val_components = memory.score_split(
        val_idx, video, actor, pair, start, actor_z, pair_z,
        global_model, pair_model, calibrators, args)
    test_components = memory.score_split(
        test_idx, video, actor, pair, start, actor_z, pair_z,
        global_model, pair_model, calibrators, args)
    weights, _ = memory.choose_weights(
        y[val_idx], val_components, (0, 1, 2, 3), args)
    return {
        "x": x, "y": y, "split": split, "names": names,
        "sample_id": sample_id, "val_idx": val_idx, "test_idx": test_idx,
        "val_score": val_components @ weights,
        "test_score": test_components @ weights,
        "weights": weights,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True)
    p.add_argument("--model-dir", default="results_chad_gpu")
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    p.add_argument("--output-dir", default="results_hybrid_fusion")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--alpha-step", type=float, default=.05)
    p.add_argument("--pr-weight", type=float, default=.70)
    # The following must match evaluate_episode_memory_v2.py.
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--memory-size", type=int, default=20)
    p.add_argument("--min-history", type=int, default=3)
    p.add_argument("--ema", type=float, default=.70)
    p.add_argument("--update-gate", type=float, default=.70)
    p.add_argument("--weight-step", type=float, default=.10)
    p.add_argument("--max-weight", type=float, default=.60)
    args = p.parse_args()

    data = np.load(args.windows, allow_pickle=False)
    prepared = prepare_memory(data, args)
    x, y, split = prepared["x"], prepared["y"], prepared["split"]
    val_idx, test_idx = prepared["val_idx"], prepared["test_idx"]
    names = prepared["names"]

    # Reproduce the exact train-only normalization used by compare_chad_npz.py.
    train_mask = split == "train"
    mean = x[train_mask].reshape(-1, x.shape[-1]).mean(0)
    std = x[train_mask].reshape(-1, x.shape[-1]).std(0)
    std[std < 1e-6] = 1.0
    normalized = np.clip(np.nan_to_num(
        ((x - mean) / std).astype(np.float32),
        nan=0, posinf=20, neginf=-20), -20, 20)
    nonpair = np.asarray([not n.startswith("pair_") for n in names])
    transformer_x = normalized[:, :, nonpair].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    memory_cdf = ValidationCDF(prepared["val_score"])
    val_memory = memory_cdf.transform(prepared["val_score"])
    test_memory = memory_cdf.transform(prepared["test_score"])

    rows = []
    score_table = pd.DataFrame({
        "sample_id": prepared["sample_id"][test_idx],
        "label": y[test_idx],
        "memory_score_raw": prepared["test_score"],
        "memory_score_percentile": test_memory,
    })
    for seed in args.seeds:
        model_file = Path(args.model_dir) / f"model_transformer_seed{seed}.pt"
        if not model_file.exists():
            raise FileNotFoundError(model_file)
        model = Transformer(transformer_x.shape[-1]).to(device)
        try:
            state = torch.load(model_file, map_location=device,
                               weights_only=True)
        except TypeError:
            state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
        val_raw = transformer_scores(
            model, transformer_x[val_idx], args.batch_size, device)
        test_raw = transformer_scores(
            model, transformer_x[test_idx], args.batch_size, device)
        tf_cdf = ValidationCDF(val_raw)
        val_tf = tf_cdf.transform(val_raw)
        test_tf = tf_cdf.transform(test_raw)

        selection = select_alpha(
            y[val_idx], val_tf, val_memory,
            args.alpha_step, args.pr_weight)
        _, alpha, hybrid_threshold, _, _ = selection
        val_hybrid = alpha * val_tf + (1 - alpha) * val_memory
        test_hybrid = alpha * test_tf + (1 - alpha) * test_memory

        transformer_threshold = best_threshold(y[val_idx], val_tf)
        memory_threshold = best_threshold(y[val_idx], val_memory)
        rows.append({"model": "transformer", "seed": seed,
                     "alpha_transformer": 1.0,
                     "alpha_memory": 0.0,
                     **result_metrics(y[test_idx], test_tf,
                                      transformer_threshold)})
        rows.append({"model": "episode_memory", "seed": seed,
                     "alpha_transformer": 0.0,
                     "alpha_memory": 1.0,
                     **result_metrics(y[test_idx], test_memory,
                                      memory_threshold)})
        rows.append({"model": "hybrid", "seed": seed,
                     "alpha_transformer": alpha,
                     "alpha_memory": 1.0 - alpha,
                     **result_metrics(y[test_idx], test_hybrid,
                                      hybrid_threshold)})
        score_table[f"transformer_seed{seed}"] = test_tf
        score_table[f"hybrid_seed{seed}"] = test_hybrid
        print("seed", seed, "alpha_transformer", alpha,
              "alpha_memory", 1 - alpha,
              result_metrics(y[test_idx], test_hybrid, hybrid_threshold))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results.to_csv(out / "hybrid_per_seed_results.csv", index=False)
    columns = ["episode_roc_auc", "episode_pr_auc", "f1",
               "precision", "recall", "alpha_transformer", "alpha_memory"]
    summary = results.groupby("model")[columns].agg(["mean", "std"])
    summary.columns = [a + "_" + b for a, b in summary.columns]
    summary.reset_index().to_csv(out / "hybrid_summary_results.csv",
                                 index=False)
    score_table.to_csv(out / "hybrid_test_scores.csv", index=False)
    with (out / "hybrid_config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args),
                   "memory_component_weights":
                   prepared["weights"].tolist(),
                   "score_calibration": "validation empirical CDF"},
                  f, ensure_ascii=False, indent=2)
    print("\nFinal summary\n", summary)
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
