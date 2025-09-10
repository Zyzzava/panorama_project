from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from .models import MatchResult, DetectResult, HomographyResult

def plot_mean_dist_bar(matches: List[MatchResult]):
    if not matches:
        return
    labels = [f"D{m.dataset}-{m.detector}-{m.pair}" for m in matches]
    values = [m.mean_dist for m in matches]
    plt.figure(figsize=(10,4))
    plt.bar(labels, values, color='steelblue')
    plt.ylabel('Mean Hamming Distance')
    plt.title('Mean Match Distance per Pair (All Datasets)')
    plt.xticks(rotation=55, ha='right')
    Path('results').mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/mean_distance_bar_all.png')
    plt.close()

def plot_distance_histograms(matches: List[MatchResult], bins=40):
    by_key = defaultdict(list)
    for m in matches:
        by_key[(m.dataset, m.detector)].extend(m.distances)
    Path('results').mkdir(exist_ok=True)
    for (ds, det), dists in by_key.items():
        if not dists:
            continue
        plt.figure(figsize=(6,4))
        plt.hist(dists, bins=bins, color='darkorange', edgecolor='black')
        plt.xlabel('Hamming Distance')
        plt.ylabel('Frequency')
        plt.title(f'Dataset {ds} - {det}')
        plt.tight_layout()
        plt.savefig(f'results/histogram_{det}_D{ds}.png')
        plt.close()

def plot_keypoints_bar(detects: Dict[Tuple[str,str], DetectResult]):
    if not detects:
        return
    entries = sorted(detects.items())
    labels = [f"D{k[0]}-{k[1]}" for k, _ in entries]
    kps = [v.total_kp for _, v in entries]
    plt.figure(figsize=(7,4))
    plt.bar(labels, kps, color='seagreen')
    plt.ylabel('Total Keypoints')
    plt.title('Keypoints per Detector per Dataset')
    plt.xticks(rotation=40, ha='right')
    Path('results').mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/keypoints_bar_all.png')
    plt.close()

def plot_match_time_bar(matches):
    # Accept both dicts and objects (e.g., MatchResult)
    def _get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    labels = []
    values = []
    for m in matches:
        ds = _get(m, 'dataset', '?')
        det = _get(m, 'detector', '?')
        pair = _get(m, 'pair', None)
        if pair is None:
            i = _get(m, 'img_i', '?')
            j = _get(m, 'img_j', '?')
            pair = f'{i}-{j}'
        t = _get(m, 'time_ms', _get(m, 'match_time_ms', _get(m, 'time', None)))
        if t is None:
            continue
        labels.append(f"D{ds}-{det}-{pair}")
        values.append(t)

    if not labels:
        return
    plt.figure(figsize=(10,4))
    plt.bar(labels, values, color='indianred')
    plt.ylabel('Matching time (ms)')
    plt.title('Descriptor Matching Time per Pair (All Datasets)')
    plt.xticks(rotation=55, ha='right')
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/match_time_bar_all.png')
    plt.close()

def plot_homography_inliers(homos: list[HomographyResult]):
    if not homos:
        return
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for h in homos:
        stats[h.dataset][h.detector][h.threshold].append(h.num_inliers)

    Path('results').mkdir(exist_ok=True)
    for dataset, det_map in stats.items():
        plt.figure(figsize=(6,4))
        for detector, thr_map in det_map.items():
            thrs = sorted(thr_map.keys())
            y = [sum(thr_map[t])/len(thr_map[t]) for t in thrs]
            plt.plot(thrs, y, marker='o', label=detector)
        plt.title(f"Dataset {dataset}: Avg Inliers vs RANSAC Threshold")
        plt.xlabel("Threshold (px)")
        plt.ylabel("Average Inliers")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/homography_inliers_D{dataset}.png")
        plt.close()

def plot_homography_time(homos: list[HomographyResult]):
    if not homos:
        return
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for h in homos:
        stats[h.dataset][h.detector][h.threshold].append(h.time_ms)

    Path('results').mkdir(exist_ok=True)
    for dataset, det_map in stats.items():
        plt.figure(figsize=(6,4))
        for detector, thr_map in det_map.items():
            thrs = sorted(thr_map.keys())
            y = [sum(thr_map[t])/len(thr_map[t]) for t in thrs]
            plt.plot(thrs, y, marker='o', label=detector)
        plt.title(f"Dataset {dataset}: Avg Homography Time vs Threshold")
        plt.xlabel("Threshold (px)")
        plt.ylabel("Time (ms)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/homography_time_D{dataset}.png")
        plt.close()

def plot_homography_summary_grid(homos: list[HomographyResult]):
    if not homos:
        return
    agg = defaultdict(lambda: defaultdict(list))
    for h in homos:
        agg[h.detector][h.threshold].append(h.num_inliers)
    detectors = sorted(agg.keys())
    thresholds = sorted({t for d in agg.values() for t in d.keys()})
    import numpy as np
    mat = []
    for det in detectors:
        row = []
        for thr in thresholds:
            vals = agg[det].get(thr, [])
            row.append(np.mean(vals) if vals else 0.0)
        mat.append(row)
    mat = np.array(mat)
    Path('results').mkdir(exist_ok=True)
    plt.figure(figsize=(6,4))
    im = plt.imshow(mat, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Avg Inliers (all datasets)")
    plt.yticks(range(len(detectors)), detectors)
    plt.xticks(range(len(thresholds)), thresholds)
    plt.xlabel("Threshold (px)")
    plt.title("Homography Inliers Summary")
    plt.tight_layout()
    plt.savefig("results/homography_inliers_summary_heatmap.png")
    plt.close()