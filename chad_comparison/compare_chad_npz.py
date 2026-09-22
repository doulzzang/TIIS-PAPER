#!/usr/bin/env python3
"""Train and compare TCN/Transformer models on prepared CHAD windows."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chad_transformer_comparison import (
    TCN, Transformer, auc, best_threshold, fit, get_metrics,
    infer, make_loader, seed_all
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True)
    p.add_argument("--models", nargs="+",
                   default=["tcn", "transformer", "actor_pair_transformer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", default="results_chad")
    args = p.parse_args()

    data = np.load(args.windows, allow_pickle=False)
    x = np.nan_to_num(data["x"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = data["y"].astype(np.int64)
    split = data["split"].astype(str)
    sample_id = data["sample_id"].astype(str)
    names = data["feature_names"].astype(str)
    pair_mask = np.array([not name.startswith("pair_") for name in names])

    train_mask = split == "train"
    mean = x[train_mask].reshape(-1, x.shape[-1]).mean(0)
    std = x[train_mask].reshape(-1, x.shape[-1]).std(0)
    std[std < 1e-6] = 1.0
    x = ((x - mean) / std).astype(np.float32)
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)

    def rows(which, use_pair):
        mask = split == which
        values = x[mask] if use_pair else x[mask][:, :, pair_mask]
        return [(sid, which, features, int(label))
                for sid, features, label
                in zip(sample_id[mask], values, y[mask])]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "train_normalization.npz", mean=mean, std=std,
             feature_names=names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    records = []
    for model_name in args.models:
        use_pair = model_name == "actor_pair_transformer"
        train, val, test = (rows(s, use_pair) for s in ("train", "val", "test"))
        for label, subset in (("train", train), ("val", val), ("test", test)):
            classes = sorted({r[3] for r in subset})
            print(model_name, label, "n=", len(subset), "classes=", classes)
            if len(classes) < 2:
                raise ValueError(label + " needs both normal and anomalous windows")
        dim = train[0][2].shape[-1]
        for seed in args.seeds:
            seed_all(seed)
            if model_name == "tcn":
                model = TCN(dim)
            elif model_name in ("transformer", "actor_pair_transformer"):
                model = Transformer(dim)
            else:
                raise ValueError("Unknown model: " + model_name)
            model = fit(model, train, val, args, device)
            vy, vs, _ = infer(model, make_loader(val, args), device)
            threshold = best_threshold(vy, vs)
            ty, ts, ids = infer(model, make_loader(test, args), device)
            result = {
                "model": model_name,
                "seed": seed,
                "episode_roc_auc": auc(ty, ts),
                "episode_pr_auc": auc(ty, ts, True),
                **{k: v for k, v in get_metrics(ty, ts, threshold).items()
                   if k not in ("roc_auc", "pr_auc")}
            }
            records.append(result)
            pd.DataFrame({"sample_id": ids, "label": ty, "score": ts}).to_csv(
                out / f"test_scores_{model_name}_seed{seed}.csv", index=False)
            torch.save(model.state_dict(),
                       out / f"model_{model_name}_seed{seed}.pt")
            print(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results = pd.DataFrame(records)
    results.to_csv(out / "per_seed_results.csv", index=False)
    metric_columns = ["episode_roc_auc", "episode_pr_auc",
                      "f1", "precision", "recall"]
    summary = results.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [a + "_" + b for a, b in summary.columns]
    summary.reset_index().to_csv(out / "summary_results.csv", index=False)
    print("\nFinal summary\n", summary)


if __name__ == "__main__":
    main()
