#!/usr/bin/env python3
"""Compact Transformer baseline for CHAD window anomaly detection.

Fair-comparison protocol:
- Uses the exact train/val/test split stored in the same NPZ used by the Agentic experiment.
- Trains only on train.
- Selects the probability threshold only on validation.
- Evaluates test once.
- Produces window-level and episode-level results compatible with the Agentic evaluation convention.

Requires: numpy, pandas, scikit-learn, torch
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score
)

EPS = 1e-12


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parsed_metadata(data):
    sample_id = data["sample_id"].astype(str)
    required = ("video_id", "actor_id", "pair_id", "start_frame", "end_frame")
    if all(k in data.files for k in required):
        return (
            sample_id,
            data["video_id"].astype(str),
            data["actor_id"].astype(str),
            data["pair_id"].astype(str),
            data["start_frame"].astype(int),
            data["end_frame"].astype(int),
        )
    parsed = [str(v).rsplit("|", 3) for v in sample_id]
    return (
        sample_id,
        np.asarray([v[0] for v in parsed]),
        np.asarray([v[1] for v in parsed]),
        np.full(len(parsed), ""),
        np.asarray([int(v[2]) for v in parsed]),
        np.asarray([int(v[3]) for v in parsed]),
    )


def fit_standardizer(x):
    # Fit feature-wise normalization from TRAIN only, pooling across time.
    flat = x.reshape(-1, x.shape[-1])
    center = np.nanmean(flat, axis=0)
    scale = np.nanstd(flat, axis=0)
    scale[scale < 1e-6] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def transform(x, center, scale):
    z = (x - center[None, None, :]) / scale[None, None, :]
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0),
                   -10.0, 10.0).astype(np.float32)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, indices):
        self.x = torch.from_numpy(x[indices]).float()
        self.y = torch.from_numpy(y[indices].astype(np.float32))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.x[i], self.y[i]


class CompactTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, layers=2,
                 dim_ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        t = x.shape[1]
        if t > self.pos.shape[1]:
            raise ValueError(f"Sequence length {t} exceeds positional limit 512")
        h = self.proj(x) + self.pos[:, :t]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))
        return self.head(h).squeeze(-1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores, labels = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(scores), np.concatenate(labels).astype(int)


def best_threshold(y, score):
    p, r, th = precision_recall_curve(y, score)
    if len(th) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + EPS)
    return float(th[int(np.argmax(f1))])


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive": int(y.sum()),
    }


def build_episodes(df, positive_col, gap_frames, score_col=None):
    rows = []
    positive = df[df[positive_col].astype(int) == 1].copy()
    if positive.empty:
        return pd.DataFrame(columns=[
            "video_id", "actor_id", "start_frame", "end_frame",
            "window_count", "max_score", "mean_score"
        ])
    for (video, actor), group in positive.groupby(
            ["video_id", "actor_id"], sort=False, dropna=False):
        group = group.sort_values(["start_frame", "end_frame"])
        cur = None
        for row in group.itertuples(index=False):
            start, end = int(row.start_frame), int(row.end_frame)
            score = float(getattr(row, score_col)) if score_col else np.nan
            if cur is None or start > cur["end_frame"] + gap_frames:
                if cur is not None:
                    rows.append(cur)
                cur = dict(video_id=str(video), actor_id=str(actor),
                           start_frame=start, end_frame=end,
                           window_count=1, scores=[score])
            else:
                cur["end_frame"] = max(cur["end_frame"], end)
                cur["window_count"] += 1
                cur["scores"].append(score)
        if cur is not None:
            rows.append(cur)
    for row in rows:
        s = np.asarray(row.pop("scores"), dtype=float)
        row["max_score"] = float(np.nanmax(s)) if np.isfinite(s).any() else np.nan
        row["mean_score"] = float(np.nanmean(s)) if np.isfinite(s).any() else np.nan
    return pd.DataFrame(rows)


def match_episodes(gt, pred, tolerance, fps):
    matches, used = [], set()
    for gi, g in gt.sort_values("start_frame").iterrows():
        c = pred[
            (pred["video_id"] == g.video_id) &
            (pred["actor_id"] == g.actor_id) &
            (pred["end_frame"] >= g.start_frame - tolerance) &
            (pred["start_frame"] <= g.end_frame + tolerance)
        ]
        c = c.loc[~c.index.isin(used)].copy()
        if c.empty:
            continue
        c["distance"] = np.maximum(
            0, np.maximum(g.start_frame - c["end_frame"],
                          c["start_frame"] - g.end_frame))
        c["onset_distance"] = abs(c["start_frame"] - g.start_frame)
        pi = c.sort_values(["distance", "onset_distance", "start_frame"]).index[0]
        p = pred.loc[pi]
        used.add(pi)
        matches.append({
            "gt_episode_index": int(gi), "pred_episode_index": int(pi),
            "video_id": g.video_id, "actor_id": g.actor_id,
            "gt_start_frame": int(g.start_frame),
            "gt_end_frame": int(g.end_frame),
            "pred_start_frame": int(p.start_frame),
            "pred_end_frame": int(p.end_frame),
            "delay_frames": int(p.start_frame - g.start_frame),
            "delay_sec": float((p.start_frame - g.start_frame) / fps),
        })
    return pd.DataFrame(matches), used


def episode_metrics(frame, gap_frames, tolerance, fps):
    gt = build_episodes(frame, "label", gap_frames)
    pred = build_episodes(frame, "prediction", gap_frames, "transformer_score")
    matches, used = match_episodes(gt, pred, tolerance, fps)
    precision = len(used) / max(len(pred), 1)
    recall = len(matches) / max(len(gt), 1)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    delay = matches["delay_sec"] if not matches.empty else pd.Series(dtype=float)
    result = {
        "gt_episodes": int(len(gt)),
        "pred_episodes": int(len(pred)),
        "matched_gt_episodes": int(len(matches)),
        "false_alarm_episodes": int(len(pred) - len(used)),
        "episode_precision": float(precision),
        "episode_recall": float(recall),
        "episode_f1": float(f1),
        "mean_delay_sec": float(delay.mean()) if len(delay) else np.nan,
        "median_delay_sec": float(delay.median()) if len(delay) else np.nan,
        "early_or_on_time_rate": float((delay <= 0).mean()) if len(delay) else np.nan,
    }
    pred = pred.copy()
    pred["matched"] = pred.index.isin(used).astype(np.int8)
    return result, gt, pred, matches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True,
                   help="Same NPZ used by evaluate_agentic_episode_validation_tuning.py")
    p.add_argument("--output-dir", default="results_transformer_baseline")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dim-ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--gap-sec", type=float, default=1.05)
    p.add_argument("--match-tolerance-sec", type=float, default=0.0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    seed_all(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.windows, allow_pickle=False)
    x = np.nan_to_num(data["x"].astype(np.float32), nan=0, posinf=0, neginf=0)
    y = data["y"].astype(int)
    split = data["split"].astype(str)
    names = data["feature_names"].astype(str)
    sample_id, video, actor, pair, start, end = parsed_metadata(data)

    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("NPZ must contain non-empty train/val/test splits.")

    # Normalization parameters are fitted on TRAIN only.
    center, scale = fit_standardizer(x[train_idx])
    xz = transform(x, center, scale)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)
    print("x shape:", x.shape, "features:", len(names))
    print("split sizes:", len(train_idx), len(val_idx), len(test_idx))
    print("positive counts:",
          int(y[train_idx].sum()), int(y[val_idx].sum()), int(y[test_idx].sum()))

    train_ds = WindowDataset(xz, y, train_idx)
    val_ds = WindowDataset(xz, y, val_idx)
    test_ds = WindowDataset(xz, y, test_idx)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False)

    model = CompactTransformer(
        x.shape[-1], args.d_model, args.nhead, args.layers,
        args.dim_ff, args.dropout).to(device)

    n_pos = max(int(y[train_idx].sum()), 1)
    n_neg = max(len(train_idx) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_pr, best_state, stale = -1.0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_n += len(yb)

        val_score, val_y = predict(model, val_loader, device)
        val_pr = float(average_precision_score(val_y, val_score))
        val_roc = float(roc_auc_score(val_y, val_score))
        row = dict(epoch=epoch, train_loss=total_loss / max(total_n, 1),
                   val_pr_auc=val_pr, val_roc_auc=val_roc)
        history.append(row)
        print(row)

        if val_pr > best_pr + 1e-8:
            best_pr = val_pr
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    # Threshold is selected on validation only.
    val_score, val_y = predict(model, val_loader, device)
    threshold = best_threshold(val_y, val_score)
    val_result = metrics(val_y, val_score, threshold)

    # Test is evaluated only after model checkpoint and threshold are fixed.
    test_score, test_y = predict(model, test_loader, device)
    test_result = metrics(test_y, test_score, threshold)

    frame = pd.DataFrame({
        "sample_id": sample_id[test_idx],
        "video_id": video[test_idx].astype(str),
        "actor_id": actor[test_idx].astype(str),
        "pair_id": pair[test_idx].astype(str),
        "start_frame": start[test_idx].astype(int),
        "end_frame": end[test_idx].astype(int),
        "label": test_y.astype(int),
        "transformer_score": test_score.astype(float),
        "prediction": (test_score >= threshold).astype(np.int8),
    })

    gap_frames = int(round(args.gap_sec * args.fps))
    tolerance = int(round(args.match_tolerance_sec * args.fps))
    ep_result, gt_ep, pred_ep, matches = episode_metrics(
        frame, gap_frames, tolerance, args.fps)

    pd.DataFrame(history).to_csv(out / "transformer_training_history.csv", index=False)
    frame.to_csv(out / "transformer_test_scores.csv", index=False)
    pd.DataFrame([{"model": "compact_transformer", **val_result}]).to_csv(
        out / "transformer_validation_results.csv", index=False)
    pd.DataFrame([{"model": "compact_transformer", **test_result}]).to_csv(
        out / "transformer_window_results.csv", index=False)
    pd.DataFrame([{"model": "compact_transformer", **ep_result}]).to_csv(
        out / "transformer_episode_results.csv", index=False)
    gt_ep.to_csv(out / "transformer_gt_episodes.csv", index=False)
    pred_ep.to_csv(out / "transformer_predicted_episodes.csv", index=False)
    matches.to_csv(out / "transformer_matched_episodes.csv", index=False)
    torch.save(best_state, out / "compact_transformer_best.pt")

    with (out / "transformer_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            **vars(args),
            "device": str(device),
            "input_shape": list(x.shape),
            "feature_names": names.tolist(),
            "train_n": int(len(train_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "validation_selected_threshold": threshold,
            "validation_metrics": val_result,
            "test_window_metrics": test_result,
            "test_episode_metrics": ep_result,
        }, f, ensure_ascii=False, indent=2)

    print("\n=== VALIDATION ===")
    print(val_result)
    print("\n=== TEST WINDOW ===")
    print(test_result)
    print("\n=== TEST EPISODE ===")
    print(ep_result)
    print("\nSaved:", out.resolve())


if __name__ == "__main__":
    main()
