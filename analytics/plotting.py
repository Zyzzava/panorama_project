from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from .models import MatchResult, DetectResult

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