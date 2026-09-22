#!/usr/bin/env python3
"""Convert official CHAD metadata ZIP into strict 2 s + 4 s windows."""
import argparse
import io
import pickle
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **_):
        return items


def read_lines(zf, name):
    return zf.read(name).decode("utf-8").split()


def split_train_val(zf, names, ratio, seed):
    groups = {0: [], 1: []}
    for name in names:
        label = np.load(io.BytesIO(zf.read("anomaly_labels/" + name + ".npy")),
                        allow_pickle=False)
        groups[int(np.any(label))].append(name)
    rng = random.Random(seed)
    train, val = [], []
    for group in groups.values():
        rng.shuffle(group)
        n = max(1, round(len(group) * ratio)) if group else 0
        val.extend(group[:n])
        train.extend(group[n:])
    return sorted(train), sorted(val)


def person_features(frame_people, width, height):
    """Return actor features from official CHAD XYWH/XYC annotations."""
    prepared = {}
    centers = {}
    for pid, value in frame_people.items():
        bbox = np.nan_to_num(np.asarray(value[0], dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
        pose = np.nan_to_num(np.asarray(value[1], dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
        if bbox.shape != (4,) or pose.shape != (17, 3):
            continue
        # Official CHAD format: bbox=[x,y,w,h], keypoints=[x,y,confidence].
        x, y, w, h = bbox
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)
        box = np.array([x / width, y / height, w / width, h / height],
                       dtype=np.float32)
        rel = pose.copy()
        rel[:, 0] = np.clip((pose[:, 0] - x) / w, -2.0, 3.0)
        rel[:, 1] = np.clip((pose[:, 1] - y) / h, -2.0, 3.0)
        rel[:, 2] = np.clip(pose[:, 2], 0.0, 1.0)
        prepared[str(pid)] = (box, rel.reshape(-1))
        centers[str(pid)] = np.array([box[0] + box[2] / 2,
                                      box[1] + box[3] / 2], dtype=np.float32)

    answer = {}
    ids = list(prepared)
    for pid in ids:
        box, pose = prepared[pid]
        pair = np.zeros(6, dtype=np.float32)
        others = [q for q in ids if q != pid]
        if others:
            nearest = min(others, key=lambda q: np.linalg.norm(centers[q] - centers[pid]))
            obox, _ = prepared[nearest]
            delta = centers[nearest] - centers[pid]
            pair[:] = [delta[0], delta[1], np.linalg.norm(delta),
                       obox[2] - box[2], obox[3] - box[3], 1.0]
        nearest_id = ""
        if others:
            nearest_id = nearest
        answer[pid] = (box, pose, pair, nearest_id)
    return answer


def video_windows(zf, name, split, args):
    with zf.open("annotations/" + name + ".pkl") as stream:
        annotations = pickle.load(stream)
    labels = np.load(io.BytesIO(zf.read("anomaly_labels/" + name + ".npy")),
                     allow_pickle=False).astype(np.int8)
    step = max(1, round(args.fps / args.sample_fps))
    obs_n = round(args.observation_sec * args.sample_fps)
    future_frames = round(args.future_sec * args.fps)
    stride_samples = max(1, round(args.stride_sec * args.sample_fps))

    # CHAD cameras 1--3 are Full-HD; camera 4 is HD.
    width, height = ((1280.0, 720.0) if str(name).startswith("4_")
                     else (args.width, args.height))
    tracks = {}
    for frame in range(0, len(labels), step):
        people = person_features(annotations.get(frame, {}), width, height)
        for pid, feature in people.items():
            tracks.setdefault(pid, []).append((frame, feature))

    rows = []
    for pid, sequence in tracks.items():
        for start in range(0, len(sequence) - obs_n + 1, stride_samples):
            window = sequence[start:start + obs_n]
            frames = np.array([item[0] for item in window])
            if np.max(np.diff(frames)) > step * args.max_gap_steps:
                continue
            end_frame = int(frames[-1])
            future_end = min(len(labels), end_frame + future_frames + 1)
            if future_end <= end_frame + 1:
                continue
            label = int(labels[end_frame + 1:future_end].max())
            features = []
            previous = None
            pair_ids = []
            for _, (box, pose, pair, nearest_id) in window:
                motion = np.zeros(4, dtype=np.float32) if previous is None else box - previous
                features.append(np.concatenate([box, motion, pose, pair]))
                previous = box
                if nearest_id:
                    pair_ids.append(str(nearest_id))
            representative_pair = (
                Counter(pair_ids).most_common(1)[0][0] if pair_ids else "")
            sample_id = name + "|" + pid + "|" + str(frames[0]) + "|" + str(end_frame)
            rows.append((np.stack(features), label, split, sample_id,
                         name, pid, representative_pair,
                         int(frames[0]), end_frame))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta-zip", required=True)
    p.add_argument("--output", default="chad_windows_2s_4s.npz")
    p.add_argument("--split-id", type=int, choices=[1, 2], default=2)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--sample-fps", type=float, default=10)
    p.add_argument("--observation-sec", type=float, default=2)
    p.add_argument("--future-sec", type=float, default=4)
    p.add_argument("--stride-sec", type=float, default=1)
    p.add_argument("--validation-ratio", type=float, default=.15)
    p.add_argument("--validation-seed", type=int, default=2026)
    p.add_argument("--width", type=float, default=1920)
    p.add_argument("--height", type=float, default=1080)
    p.add_argument("--max-gap-steps", type=int, default=2)
    p.add_argument("--max-videos", type=int, default=0,
                   help="0 means all; positive values are for a quick test")
    args = p.parse_args()

    if args.split_id == 1:
        raise ValueError("split 1 train contains normal videos only. "
                         "Use split 2 for supervised TCN/Transformer comparison.")
    with zipfile.ZipFile(args.meta_zip) as zf:
        official_train = read_lines(zf, "splits/train_split_2.txt")
        test = read_lines(zf, "splits/test_split_2.txt")
        train, val = split_train_val(
            zf, official_train, args.validation_ratio, args.validation_seed)
        plan = [(x, "train") for x in train]
        plan += [(x, "val") for x in val]
        plan += [(x, "test") for x in test]
        if args.max_videos:
            # Select from every split so the resulting file remains testable.
            chosen = []
            for split in ("train", "val", "test"):
                pool = [x for x in plan if x[1] == split]
                normal = [x for x in pool if x[0].endswith("_0")]
                anomaly = [x for x in pool if x[0].endswith("_1")]
                half = max(1, args.max_videos // 2)
                selected = normal[:half] + anomaly[:max(1, args.max_videos - half)]
                chosen += selected[:args.max_videos]
            plan = chosen
        rows = []
        for name, split in tqdm(plan, desc="CHAD videos"):
            rows.extend(video_windows(zf, name, split, args))

    if not rows:
        raise RuntimeError("No valid windows were generated")
    x = np.stack([r[0] for r in rows]).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray([r[1] for r in rows], dtype=np.int8)
    split = np.asarray([r[2] for r in rows])
    sample_id = np.asarray([r[3] for r in rows])
    video_id = np.asarray([r[4] for r in rows])
    actor_id = np.asarray([r[5] for r in rows])
    pair_id = np.asarray([r[6] for r in rows])
    start_frame = np.asarray([r[7] for r in rows], dtype=np.int32)
    end_frame = np.asarray([r[8] for r in rows], dtype=np.int32)
    feature_names = (
        ["bbox_x", "bbox_y", "bbox_w", "bbox_h",
         "d_bbox_x", "d_bbox_y", "d_bbox_w", "d_bbox_h"]
        + [f"pose_{j}_{axis}" for j in range(17) for axis in ("x", "y", "c")]
        + ["pair_dx", "pair_dy", "pair_dist", "pair_dw", "pair_dh", "pair_present"]
    )
    np.savez_compressed(args.output, x=x, y=y, split=split,
                        sample_id=sample_id,
                        video_id=video_id, actor_id=actor_id, pair_id=pair_id,
                        start_frame=start_frame, end_frame=end_frame,
                        feature_names=np.asarray(feature_names))
    print("Saved:", Path(args.output).resolve())
    for s in ("train", "val", "test"):
        mask = split == s
        print(s, "windows=", int(mask.sum()), "positive=", int(y[mask].sum()))


if __name__ == "__main__":
    main()
