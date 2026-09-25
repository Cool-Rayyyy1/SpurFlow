#!/usr/bin/env python3
"""ARI between per-step mixture dominant_k (argmax pi head) and GEdit edit_type.

Reads {output_dir}/manifest.json for stem -> edit_type and
{output_dir}/mixture_stats/{stem}.json (written by run_editflow_gedit_infer.py
--dump_mixture_stats) for per-NFE-step dominant_k. Composite / reference-based
edit types are excluded by default so that #classes <= K (16 for k16 models).

Outputs {output_dir}/pi_ari_summary.json and prints a per-step report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

# GEdit-v2 composite / reference / open-ended types: no single dominant edit
# operation, so they are not expected to map to one mixture head.
DEFAULT_EXCLUDE_TYPES = (
    "hybrid",
    "openset",
    "chart_editing",
    "character_reference",
    "object_reference",
    "style_reference",
    "line2image",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Student output dir containing manifest.json and mixture_stats/.")
    p.add_argument(
        "--exclude_types", type=str, default=",".join(DEFAULT_EXCLUDE_TYPES),
        help="Comma-separated edit types to drop (default: composite/reference types).")
    p.add_argument(
        "--keep_types", type=str, default="",
        help="Comma-separated whitelist; if set, overrides --exclude_types.")
    p.add_argument(
        "--out_json", type=Path, default=None,
        help="Summary path (default: {output_dir}/pi_ari_summary.json).")
    p.add_argument(
        "--labels_json", type=Path, default=None,
        help="Optional {key: {edit_type: ...}} map (e.g. imgedit annotations/"
        "basic_edit.json) supplying labels for stats files missing from "
        "manifest.json (interrupted/resumed runs).")
    p.add_argument(
        "--soft", action="store_true",
        help="Also run soft analyses on the 16-dim mean_weights vectors: "
        "z-score + k-means ARI/NMI, and a leave-one-out kNN probe accuracy. "
        "Picks up weak type signal that the hard dominant_k argmax erases.")
    return p.parse_args()


def comb2(x: int) -> int:
    return x * (x - 1) // 2


def adjusted_rand_index(labels_true: Sequence, labels_pred: Sequence) -> float:
    n = len(labels_true)
    if n < 2:
        return 0.0
    cont: Dict = defaultdict(Counter)
    for t, p in zip(labels_true, labels_pred):
        cont[t][p] += 1
    sum_ij = sum(comb2(v) for row in cont.values() for v in row.values())
    sum_a = sum(comb2(sum(row.values())) for row in cont.values())
    sum_b = sum(comb2(v) for v in Counter(labels_pred).values())
    total = comb2(n)
    expected = sum_a * sum_b / total
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        return 0.0
    return (sum_ij - expected) / denom


def normalized_mutual_info(labels_true: Sequence, labels_pred: Sequence) -> float:
    import math

    n = len(labels_true)
    if n == 0:
        return 0.0
    joint = Counter(zip(labels_true, labels_pred))
    ct, cp = Counter(labels_true), Counter(labels_pred)
    mi = 0.0
    for (t, p), c in joint.items():
        mi += (c / n) * math.log(c * n / (ct[t] * cp[p]))
    h_t = -sum((c / n) * math.log(c / n) for c in ct.values())
    h_p = -sum((c / n) * math.log(c / n) for c in cp.values())
    denom = math.sqrt(h_t * h_p)
    return mi / denom if denom > 0 else 0.0


def load_samples(output_dir: Path, keep, exclude, labels_json: Path = None) -> List[Dict]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    stats_dir = output_dir / "mixture_stats"

    # stem -> edit_type from the manifest, plus optional external labels for
    # stats files a resumed/interrupted run never re-registered.
    label_map: Dict[str, str] = {}
    for stem, rec in manifest.items():
        if "suite" in rec:
            if rec.get("suite") != "basic":
                continue
            stem = str(rec.get("key", stem))
        label_map[stem] = str(rec.get("edit_type", "unknown"))
    if labels_json is not None:
        extra = json.loads(labels_json.read_text(encoding="utf-8"))
        for key, item in extra.items():
            if isinstance(item, dict):
                label_map.setdefault(str(key), str(item.get("edit_type", "unknown")))

    samples, unlabeled = [], 0
    for stats_path in sorted(stats_dir.glob("*.json")):
        stem = stats_path.stem
        edit_type = label_map.get(stem)
        if edit_type is None:
            unlabeled += 1
            continue
        if keep:
            if edit_type not in keep:
                continue
        elif edit_type in exclude:
            continue
        steps = json.loads(stats_path.read_text(encoding="utf-8")).get("steps", [])
        dominant = [int(s.get("dominant_k", -1)) for s in steps]
        mean_w = [list(map(float, s.get("mean_weights", []))) for s in steps]
        if dominant:
            samples.append({
                "stem": stem,
                "edit_type": edit_type,
                "dominant_k": dominant,
                "mean_weights": mean_w,
            })
    if unlabeled:
        print(f"[warn] {unlabeled} stats files have no edit_type label "
              "(not in manifest; pass --labels_json to cover them)")
    return samples


def contingency(labels_true: Sequence, labels_pred: Sequence) -> Dict[str, Dict[str, int]]:
    table: Dict = defaultdict(Counter)
    for t, p in zip(labels_true, labels_pred):
        table[t][str(p)] += 1
    return {t: dict(sorted(row.items(), key=lambda kv: -kv[1])) for t, row in sorted(table.items())}


def _kmeans(X, k: int, iters: int = 200, restarts: int = 20, seed: int = 0):
    """Plain k-means (numpy only); returns labels of the best-inertia restart."""
    import numpy as np

    rng = np.random.default_rng(seed)
    best_inertia, best_labels = None, None
    for _ in range(restarts):
        centers = X[rng.choice(len(X), size=k, replace=False)].copy()
        labels = None
        for _ in range(iters):
            dist = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
            labels = dist.argmin(1)
            new_centers = np.stack([
                X[labels == j].mean(0) if (labels == j).any() else centers[j]
                for j in range(k)
            ])
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def _knn_loo_accuracy(X, labels: List[str], k: int = 20) -> float:
    """Leave-one-out kNN probe (cosine similarity)."""
    import numpy as np

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    nbr = np.argsort(-sim, axis=1)[:, :k]
    correct = 0
    for i in range(len(labels)):
        votes = Counter(labels[j] for j in nbr[i])
        correct += votes.most_common(1)[0][0] == labels[i]
    return correct / len(labels)


def soft_analysis(samples: List[Dict], num_steps: int, seed: int = 0) -> List[Dict]:
    """Cluster the soft 16-dim mean_weights (no argmax anywhere).

    Per feature set (step0 / step1 / both concatenated): z-score each dim
    across samples (removes the global content-independent pi profile),
    k-means with k = #edit_types -> ARI/NMI, plus a LOO kNN probe accuracy
    against the majority-class baseline.
    """
    import numpy as np

    full = [s for s in samples if len(s["mean_weights"]) >= num_steps
            and all(len(w) > 0 for w in s["mean_weights"][:num_steps])]
    labels = [s["edit_type"] for s in full]
    n_types = len(set(labels))
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    feature_sets = {f"step{i}": np.array([s["mean_weights"][i] for s in full])
                    for i in range(num_steps)}
    if num_steps > 1:
        feature_sets["both"] = np.concatenate(
            [feature_sets[f"step{i}"] for i in range(num_steps)], axis=1)

    results = []
    for name, X in feature_sets.items():
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
        km_labels = _kmeans(Xz, k=n_types, seed=seed)
        ari = adjusted_rand_index(labels, [int(v) for v in km_labels])
        nmi = normalized_mutual_info(labels, [int(v) for v in km_labels])
        knn_acc = _knn_loo_accuracy(Xz, labels)
        results.append({
            "features": name,
            "num_samples": len(full),
            "kmeans_k": n_types,
            "kmeans_ari": ari,
            "kmeans_nmi": nmi,
            "knn_loo_accuracy": knn_acc,
            "majority_baseline": majority,
        })
        print(
            f"[soft] {name:6s}  kmeans ARI: {ari:.4f}  NMI: {nmi:.4f}  "
            f"| kNN probe acc: {knn_acc:.4f} (baseline {majority:.4f})")
    return results


def main() -> None:
    args = parse_args()
    keep = {s.strip() for s in args.keep_types.split(",") if s.strip()}
    exclude = {s.strip() for s in args.exclude_types.split(",") if s.strip()}
    samples = load_samples(args.output_dir, keep, exclude, labels_json=args.labels_json)
    if not samples:
        raise SystemExit(f"No samples with mixture stats under {args.output_dir}")

    num_steps = max(len(s["dominant_k"]) for s in samples)
    types = sorted({s["edit_type"] for s in samples})
    print(f"[pi-ari] {len(samples)} samples, {len(types)} edit types, {num_steps} NFE steps")
    print(f"[pi-ari] types: {', '.join(types)}")

    summary = {
        "output_dir": str(args.output_dir),
        "num_samples": len(samples),
        "edit_types": types,
        "excluded_types": sorted(exclude) if not keep else [],
        "kept_types": sorted(keep),
        "steps": [],
    }
    for step in range(num_steps):
        pairs = [
            (s["edit_type"], s["dominant_k"][step])
            for s in samples if len(s["dominant_k"]) > step
        ]
        labels_true = [t for t, _ in pairs]
        labels_pred = [p for _, p in pairs]
        ari = adjusted_rand_index(labels_true, labels_pred)
        nmi = normalized_mutual_info(labels_true, labels_pred)
        head_hist = dict(sorted(Counter(labels_pred).items()))
        table = contingency(labels_true, labels_pred)
        summary["steps"].append({
            "step": step,
            "num_samples": len(pairs),
            "ari": ari,
            "nmi": nmi,
            "num_heads_used": len(head_hist),
            "head_histogram": {str(k): v for k, v in head_hist.items()},
            "contingency": table,
        })
        print(f"\n== step {step}  (n={len(pairs)}) ==")
        print(f"ARI: {ari:.4f}   NMI: {nmi:.4f}   heads used: {len(head_hist)}")
        print(f"head histogram: {head_hist}")
        print("edit_type -> dominant_k counts:")
        for t, row in table.items():
            print(f"  {t}: {row}")

    out_json = args.out_json or (args.output_dir / "pi_ari_summary.json")
    if args.soft:
        print("\n[soft] mean_weights (no argmax): z-score + k-means / kNN probe")
        summary["soft"] = soft_analysis(samples, num_steps)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[pi-ari] summary -> {out_json}")


if __name__ == "__main__":
    main()
