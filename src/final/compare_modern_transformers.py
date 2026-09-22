#!/usr/bin/env python3
"""Fair modern time-series Transformer comparison on prepared CHAD windows.

Models
------
* tcn: temporal convolution baseline
* transformer: vanilla temporal-token Transformer
* actor_pair_transformer: vanilla Transformer with pair features
* patchtst: channel-independent temporal patch Transformer classifier
* itransformer: variate-token (inverted) Transformer classifier

The same prepared NPZ, official split, training loop, validation-selected
threshold, seeds, and test windows are used for every model.  Under the
default ``legacy`` feature policy, TCN/Transformer/PatchTST/iTransformer use
the same non-pair features, while actor_pair_transformer alone receives the
explicit pair features.  Use ``--feature-policy all`` to give every model all
features; in that mode transformer and actor_pair_transformer are redundant.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from chad_transformer_comparison import (
    PositionalEncoding, TCN, Transformer, auc, best_threshold, fit,
    get_metrics, infer, make_loader, seed_all,
)


class PatchTSTClassifier(nn.Module):
    """Compact channel-independent PatchTST adaptation for binary labels."""

    def __init__(self, dim, seq_len, patch_len=4, patch_stride=2,
                 d_model=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        if patch_len > seq_len:
            patch_len = seq_len
        if patch_stride > patch_len:
            patch_stride = patch_len
        self.dim = dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_count = 1 + (seq_len - patch_len) // patch_stride
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = PositionalEncoding(d_model, max_len=self.patch_count + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, time, variables] -> [batch, variables, patches, patch_len]
        batch, _, channels = x.shape
        patches = x.transpose(1, 2).unfold(
            dimension=2, size=self.patch_len, step=self.patch_stride
        )
        patches = patches.contiguous().view(
            batch * channels, self.patch_count, self.patch_len
        )
        h = self.encoder(self.pos(self.patch_embed(patches))).mean(dim=1)
        # Shared channel-independent encoder; pool variable representations.
        h = h.view(batch, channels, -1).mean(dim=1)
        return self.head(self.norm(h)).squeeze(-1)


class ITransformerClassifier(nn.Module):
    """Compact iTransformer adaptation using variables as attention tokens."""

    def __init__(self, dim, seq_len, d_model=64, heads=4, layers=2,
                 dropout=0.1):
        super().__init__()
        self.time_embed = nn.Linear(seq_len, d_model)
        # A learned variable embedding keeps the token identity explicit.
        self.variable_embed = nn.Parameter(torch.zeros(1, dim, d_model))
        nn.init.normal_(self.variable_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # Each feature variable becomes one token containing its time history.
        h = self.time_embed(x.transpose(1, 2)) + self.variable_embed
        h = self.encoder(h).mean(dim=1)
        return self.head(self.norm(h)).squeeze(-1)


def build_model(name, dim, seq_len, args):
    if name == "tcn":
        return TCN(dim, hidden=args.d_model)
    if name in ("transformer", "actor_pair_transformer"):
        return Transformer(
            dim, d_model=args.d_model, heads=args.heads, layers=args.layers
        )
    if name == "patchtst":
        return PatchTSTClassifier(
            dim, seq_len, patch_len=args.patch_len,
            patch_stride=args.patch_stride, d_model=args.d_model,
            heads=args.heads, layers=args.layers,
        )
    if name == "itransformer":
        return ITransformerClassifier(
            dim, seq_len, d_model=args.d_model,
            heads=args.heads, layers=args.layers,
        )
    raise ValueError("Unknown model: " + name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True)
    p.add_argument("--models", nargs="+", default=[
        "tcn", "transformer", "actor_pair_transformer",
        "patchtst", "itransformer",
    ])
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--patch-len", type=int, default=4)
    p.add_argument("--patch-stride", type=int, default=2)
    p.add_argument("--feature-policy", choices=["legacy", "all"],
                   default="legacy")
    p.add_argument("--output-dir", default="results_modern_transformers")
    args = p.parse_args()

    valid_models = {
        "tcn", "transformer", "actor_pair_transformer",
        "patchtst", "itransformer",
    }
    unknown = set(args.models) - valid_models
    if unknown:
        p.error("Unknown model(s): " + ", ".join(sorted(unknown)))
    if args.d_model % args.heads:
        p.error("--d-model must be divisible by --heads")

    data = np.load(args.windows, allow_pickle=False)
    required = {"x", "y", "split", "sample_id", "feature_names"}
    missing = required - set(data.files)
    if missing:
        raise ValueError("NPZ missing arrays: " + str(sorted(missing)))
    x = np.nan_to_num(data["x"].astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = data["y"].astype(np.int64)
    split = data["split"].astype(str)
    sample_id = data["sample_id"].astype(str)
    names = data["feature_names"].astype(str)
    if x.ndim != 3:
        raise ValueError(f"Expected x=[N,T,C], got {x.shape}")
    if not {"train", "val", "test"}.issubset(set(split)):
        raise ValueError("split must contain train, val, and test")

    # True means a base/non-pair input feature.  Pair features are reserved for
    # actor_pair_transformer under the legacy policy.
    base_mask = np.array([not name.startswith("pair_") for name in names])
    train_mask = split == "train"
    mean = x[train_mask].reshape(-1, x.shape[-1]).mean(0)
    std = x[train_mask].reshape(-1, x.shape[-1]).std(0)
    std[std < 1e-6] = 1.0
    x = ((x - mean) / std).astype(np.float32)
    # Protect mixed-precision training from rare extreme metadata values.
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0),
                -20.0, 20.0)

    def model_uses_all_features(model_name):
        return (args.feature_policy == "all" or
                model_name == "actor_pair_transformer")

    def rows(which, use_all):
        mask = split == which
        values = x[mask] if use_all else x[mask][:, :, base_mask]
        return [(sid, which, features, int(label))
                for sid, features, label
                in zip(sample_id[mask], values, y[mask])]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "train_normalization.npz", mean=mean, std=std,
             feature_names=names, base_feature_mask=base_mask)
    config = vars(args).copy()
    config.update({
        "window_shape": list(x.shape),
        "split_counts": {s: int(np.sum(split == s))
                         for s in ("train", "val", "test")},
        "positive_counts": {s: int(y[split == s].sum())
                            for s in ("train", "val", "test")},
        "base_feature_count": int(base_mask.sum()),
        "all_feature_count": int(len(names)),
    })
    (out / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Samples:", config["split_counts"],
          "positive:", config["positive_counts"])

    records = []
    for model_name in args.models:
        use_all = model_uses_all_features(model_name)
        train, val, test = (rows(s, use_all)
                            for s in ("train", "val", "test"))
        for label, subset in (("train", train), ("val", val), ("test", test)):
            classes = sorted({r[3] for r in subset})
            print(model_name, label, "n=", len(subset), "classes=", classes)
            if len(classes) < 2:
                raise ValueError(label + " needs both normal and anomalous windows")
        seq_len, dim = train[0][2].shape
        feature_set = "all" if use_all else "base_without_pair"

        for seed in args.seeds:
            seed_all(seed)
            model = build_model(model_name, dim, seq_len, args)
            parameters = sum(p.numel() for p in model.parameters())
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            train_start = time.perf_counter()
            model = fit(model, train, val, args, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds = time.perf_counter() - train_start

            vy, vs, _ = infer(model, make_loader(val, args), device)
            threshold = best_threshold(vy, vs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            ty, ts, ids = infer(model, make_loader(test, args), device)
            if device.type == "cuda":
                torch.cuda.synchronize()
                peak_memory_mb = torch.cuda.max_memory_allocated() / 2**20
            else:
                peak_memory_mb = float("nan")
            infer_seconds = time.perf_counter() - infer_start

            result = {
                "model": model_name,
                "seed": seed,
                "feature_set": feature_set,
                "parameters": parameters,
                "train_seconds": train_seconds,
                "inference_ms_per_window": 1000.0 * infer_seconds / len(test),
                "peak_gpu_memory_mb": peak_memory_mb,
                "episode_roc_auc": auc(ty, ts),
                "episode_pr_auc": auc(ty, ts, True),
                **{k: v for k, v in get_metrics(ty, ts, threshold).items()
                   if k not in ("roc_auc", "pr_auc")},
            }
            records.append(result)
            pd.DataFrame({"sample_id": ids, "label": ty, "score": ts}).to_csv(
                out / f"test_scores_{model_name}_seed{seed}.csv", index=False
            )
            torch.save(model.state_dict(),
                       out / f"model_{model_name}_seed{seed}.pt")
            print(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results = pd.DataFrame(records)
    results.to_csv(out / "per_seed_results.csv", index=False)
    metric_columns = [
        "episode_roc_auc", "episode_pr_auc", "f1", "precision", "recall",
        "parameters", "train_seconds", "inference_ms_per_window",
        "peak_gpu_memory_mb",
    ]
    summary = results.groupby(["model", "feature_set"])[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [a + "_" + b for a, b in summary.columns]
    summary.reset_index().to_csv(out / "summary_results.csv", index=False)
    print("\nFinal summary\n", summary)


if __name__ == "__main__":
    main()
