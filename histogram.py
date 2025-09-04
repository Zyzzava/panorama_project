import csv
import matplotlib.pyplot as plt
from statistics import mean, median, pstdev
from collections import defaultdict
from pathlib import Path
from glob import glob

def read_results(path, dataset_tag):
    matches = []
    detect = {}
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            t = row['type']
            if t == 'match':
                distances = []
                if row.get('distances'):
                    distances = [float(x) for x in row['distances'].split(';') if x]
                matches.append({
                    'dataset': dataset_tag,
                    'detector': row['detector'],
                    'pair': f"{row['img_i']}-{row['img_j']}",
                    'num_matches': int(row['num_matches']),
                    'mean_dist': float(row['mean_dist']),
                    'time_ms': float(row['time_ms']),
                    'distances': distances
                })
            elif t == 'detect':
                total_kp = int(row['num_matches']) if row['num_matches'] else 0
                detect_time = float(row['time_ms']) if row['time_ms'] else 0.0
                detect[(dataset_tag, row['detector'])] = {
                    'dataset': dataset_tag,
                    'total_kp': total_kp,
                    'detect_time_ms': detect_time
                }
    return matches, detect

def load_all():
    all_matches = []
    detect_map = {}
    for path in sorted(glob("results/results*.txt")):
        # Extract number for dataset tag (fallback to filename)
        tag = path.split("results")[-1].split(".")[0] or ""
        tag = tag if tag else "?"
        m, d = read_results(path, tag)
        all_matches.extend(m)
        detect_map.update(d)
    return all_matches, detect_map

def summarize_matches(matches):
    by_key = defaultdict(list)  # (dataset, detector)
    for m in matches:
        by_key[(m['dataset'], m['detector'])].append(m)
    out = []
    for (ds, det), lst in by_key.items():
        mdists = [x['mean_dist'] for x in lst]
        nm = [x['num_matches'] for x in lst]
        mt = [x['time_ms'] for x in lst]
        out.append({
            'dataset': ds,
            'detector': det,
            'pairs': len(lst),
            'avg_mean_dist': mean(mdists),
            'median_mean_dist': median(mdists),
            'std_mean_dist': pstdev(mdists) if len(mdists) > 1 else 0.0,
            'total_matches': sum(nm),
            'avg_matches_per_pair': mean(nm),
            'avg_match_time_ms': mean(mt),
            'total_match_time_ms': sum(mt)
        })
    return out

def plot_mean_dist_bar(matches):
    labels = [f"D{m['dataset']}-{m['detector']}-{m['pair']}" for m in matches]
    values = [m['mean_dist'] for m in matches]
    if not labels:
        return
    plt.figure(figsize=(10,4))
    plt.bar(labels, values, color='steelblue')
    plt.ylabel('Mean Hamming Distance')
    plt.title('Mean Match Distance per Pair (All Datasets)')
    plt.xticks(rotation=55, ha='right')
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/mean_distance_bar_all.png')
    plt.close()

def plot_distance_histograms(matches, bins=40):
    by_det_ds = defaultdict(list)  # (dataset, detector)
    for m in matches:
        by_det_ds[(m['dataset'], m['detector'])].extend(m['distances'])
    Path('results').mkdir(exist_ok=True)
    for (ds, det), dists in by_det_ds.items():
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

def plot_keypoints_bar(detect_summary):
    if not detect_summary:
        return
    entries = sorted(detect_summary.items())
    labels = [f"D{k[0]}-{k[1]}" for k, _ in entries]
    kps = [v['total_kp'] for _, v in entries]
    plt.figure(figsize=(7,4))
    plt.bar(labels, kps, color='seagreen')
    plt.ylabel('Total Keypoints')
    plt.title('Keypoints per Detector per Dataset')
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/keypoints_bar_all.png')
    plt.close()

def print_overview(detect_summary, match_summary):
    print("=== Overview (per dataset & detector) ===")
    ms_key = {(m['dataset'], m['detector']): m for m in match_summary}
    for key, info in sorted(detect_summary.items()):
        ds, det = key
        ms = ms_key.get(key, {})
        print(f"D{ds}-{det}: total_kp={info['total_kp']}, "
              f"detect_time_ms={info['detect_time_ms']:.2f}, "
              f"pairs={ms.get('pairs','-')}, "
              f"avg_match_time_ms={ms.get('avg_match_time_ms','-')}, "
              f"avg_mean_dist={ms.get('avg_mean_dist','-')}")

if __name__ == "__main__":
    matches, detect_summary = load_all()
    match_summary = summarize_matches(matches)
    plot_mean_dist_bar(matches)
    plot_distance_histograms(matches)
    plot_keypoints_bar(detect_summary)
    print_overview(detect_summary, match_summary)