#!/usr/bin/env python3
"""Fair TCN/Transformer comparison on CHAD person metadata.

Input: one CSV row per (video, frame, person). Models see only the observation
window; the following future window is used solely to construct the label.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REQUIRED = {"video_id", "frame", "person_id", "label", "split",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h"}
BASE = ["bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "d_bbox_x", "d_bbox_y", "d_bbox_w", "d_bbox_h"]
PAIR = ["pair_dx", "pair_dy", "pair_dist", "pair_dw", "pair_dh", "pair_present"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auc(y, score, pr=False):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score) if pr else roc_auc_score(y, score))


def best_threshold(y, score):
    q = np.unique(np.quantile(score, np.linspace(.01, .99, 199)))
    f = [f1_score(y, score >= t, zero_division=0) for t in q]
    return float(q[int(np.argmax(f))])


def get_metrics(y, score, threshold):
    pred = score >= threshold
    return {
        "roc_auc": auc(y, score),
        "pr_auc": auc(y, score, True),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": threshold, "n": len(y), "positive": int(y.sum())
    }


def prepare_frame_table(df):
    df = df.sort_values(["video_id", "person_id", "frame"]).copy()
    for c in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        df["d_" + c] = df.groupby(["video_id", "person_id"])[c].diff().fillna(0)
    df[PAIR] = 0.0
    for _, idx in tqdm(df.groupby(["video_id", "frame"], sort=False).groups.items(),
                       desc="Pair features"):
        ids = np.asarray(list(idx))
        if len(ids) < 2:
            continue
        xy = df.loc[ids, ["bbox_x", "bbox_y"]].to_numpy(float)
        dist = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        near = dist.argmin(1)
        other = ids[near]
        cur = df.loc[ids]
        oth = df.loc[other]
        df.loc[ids, "pair_dx"] = oth.bbox_x.to_numpy() - cur.bbox_x.to_numpy()
        df.loc[ids, "pair_dy"] = oth.bbox_y.to_numpy() - cur.bbox_y.to_numpy()
        df.loc[ids, "pair_dist"] = dist[np.arange(len(ids)), near]
        df.loc[ids, "pair_dw"] = oth.bbox_w.to_numpy() - cur.bbox_w.to_numpy()
        df.loc[ids, "pair_dh"] = oth.bbox_h.to_numpy() - cur.bbox_h.to_numpy()
        df.loc[ids, "pair_present"] = 1.0
    return df


def build_windows(df, features, fps, sample_fps, obs_sec, future_sec, stride_sec):
    step = max(1, round(fps / sample_fps))
    obs_n, fut_n = round(sample_fps * obs_sec), round(sample_fps * future_sec)
    stride = max(1, round(sample_fps * stride_sec))
    rows = []
    groups = df.groupby(["video_id", "person_id"], sort=False)
    for (video, person), g in tqdm(groups, desc="Windows"):
        g = g.sort_values("frame").iloc[::step].reset_index(drop=True)
        frames = g.frame.to_numpy()
        for start in range(0, len(g) - obs_n - fut_n + 1, stride):
            end, future_end = start + obs_n, start + obs_n + fut_n
            if np.max(np.diff(frames[start:future_end])) > step * 2:
                continue
            x = g.loc[start:end - 1, features].to_numpy(np.float32)
            y = int(g.loc[end:future_end - 1, "label"].astype(int).max())
            split = str(g.loc[start, "split"]).lower()
            sid = f"{video}|{person}|{frames[start]}|{frames[end-1]}"
            rows.append((sid, split, x, y))
    if not rows:
        raise ValueError("No windows: check fps, sequence length, and tracking gaps")
    return rows


class Windows(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        sid, _, x, y = self.rows[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), sid


class TCN(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(dim, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(.1),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.head(self.body(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, dim, d_model=64, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 2, .1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
    def forward(self, x):
        h = self.encoder(self.pos(self.embed(x)))
        return self.head(h.mean(1)).squeeze(-1)


def make_loader(rows, args, shuffle=False):
    return DataLoader(Windows(rows), args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=True)


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ys, scores, ids = [], [], []
    for x, y, sid in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            z = model(x.to(device, non_blocking=True))
        ys.extend(y.numpy().astype(int))
        scores.extend(torch.sigmoid(z).float().cpu().numpy())
        ids.extend(sid)
    return np.asarray(ys), np.asarray(scores), ids


def fit(model, train, val, args, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    positive = sum(x[3] for x in train)
    weight = (len(train) - positive) / max(1, positive)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1, weight), device=device))
    best, state, stale = -float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        for x, y, _ in make_loader(train, args, True):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(opt)
            scaler.update()
        vy, vs, _ = infer(model, make_loader(val, args), device)
        value = auc(vy, vs, True)
        if np.isnan(value):
            raise ValueError("Validation split needs both normal and anomalous samples")
        if value > best + 1e-5:
            best, stale = value, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    return model


def external_scores(specs, rows, records):
    label = {x[0]: x[3] for x in rows}
    for spec in specs:
        name, filename = spec.split("=", 1)
        df = pd.read_csv(filename)
        if not {"sample_id", "score", "split"}.issubset(df.columns):
            raise ValueError("External score CSV needs sample_id, score, split")
        df = df[df.sample_id.isin(label)].copy()
        val = df[df.split.str.lower().eq("val")]
        test = df[df.split.str.lower().eq("test")]
        vy = np.asarray([label[x] for x in val.sample_id])
        ty = np.asarray([label[x] for x in test.sample_id])
        threshold = best_threshold(vy, val.score.to_numpy())
        records.append({"model": name, "seed": "external",
                        **get_metrics(ty, test.score.to_numpy(), threshold)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--sample-fps", type=float, default=10)
    p.add_argument("--observation-sec", type=float, default=2)
    p.add_argument("--future-sec", type=float, default=4)
    p.add_argument("--stride-sec", type=float, default=1)
    p.add_argument("--models", nargs="+",
                   default=["tcn", "transformer", "actor_pair_transformer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", default="results_chad")
    p.add_argument("--external-score", action="append", default=[])
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    missing = REQUIRED - set(df)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    df["split"] = df.split.astype(str).str.lower()
    if not {"train", "val", "test"}.issubset(set(df.split)):
        raise ValueError("split must contain train, val, and test")
    df = prepare_frame_table(df)
    pose = sorted(c for c in df if c.startswith("pose_"))
    features = BASE + pose + PAIR
    rows = build_windows(df, features, args.fps, args.sample_fps,
                         args.observation_sec, args.future_sec, args.stride_sec)
    splits = {s: [x for x in rows if x[1] == s] for s in ("train", "val", "test")}
    if any(not x for x in splits.values()):
        raise ValueError({k: len(v) for k, v in splits.items()})

    stack = np.concatenate([x[2] for x in splits["train"]])
    mean, std = stack.mean(0), stack.std(0)
    std[std < 1e-6] = 1
    def normalized(items):
        return [(sid, sp, ((x - mean) / std).astype(np.float32), y)
                for sid, sp, x, y in items]
    splits = {k: normalized(v) for k, v in splits.items()}
    np.savez(out / "train_normalization.npz", mean=mean, std=std,
             feature_cols=np.asarray(features))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "Samples:", json.dumps({k: len(v) for k, v in splits.items()}))
    records = []
    for seed in args.seeds:
        for name in args.models:
            seed_all(seed)
            use_pair = name == "actor_pair_transformer"
            dim = len(features) if use_pair else len(BASE + pose)
            selected = splits if use_pair else {
                k: [(sid, sp, x[:, :dim], y) for sid, sp, x, y in v]
                for k, v in splits.items()}
            if name == "tcn":
                model = TCN(dim)
            elif name in {"transformer", "actor_pair_transformer"}:
                model = Transformer(dim)
            else:
                raise ValueError("Unknown model: " + name)
            model = fit(model, selected["train"], selected["val"], args, device)
            vy, vs, _ = infer(model, make_loader(selected["val"], args), device)
            threshold = best_threshold(vy, vs)
            ty, ts, ids = infer(model, make_loader(selected["test"], args), device)
            rec = {"model": name, "seed": seed, **get_metrics(ty, ts, threshold)}
            records.append(rec)
            pd.DataFrame({"sample_id": ids, "label": ty, "score": ts}).to_csv(
                out / f"test_scores_{name}_seed{seed}.csv", index=False)
            print(rec)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    external_scores(args.external_score, sum(splits.values(), []), records)
    result = pd.DataFrame(records)
    result.to_csv(out / "per_seed_results.csv", index=False)
    cols = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    base = result[result.seed != "external"].groupby("model")[cols].agg(["mean", "std"])
    base.columns = [a + "_" + b for a, b in base.columns]
    base.reset_index().to_csv(out / "summary_results.csv", index=False)
    print(base)


if __name__ == "__main__":
    main()
