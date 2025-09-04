import csv
import matplotlib.pyplot as plt
from statistics import mean, median, pstdev

def read_matches(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            if row['type'] == 'match':
                rows.append({
                    'detector': row['detector'],
                    'pair': f"{row['img_i']}-{row['img_j']}",
                    'num_matches': int(row['num_matches']),
                    'mean_dist': float(row['mean_dist']),
                    'time_ms': float(row['time_ms'])
                })
    return rows

def summarize(rows):
    by_det = {}
    for r in rows:
        by_det.setdefault(r['detector'], []).append(r)
    summary = []
    for det, lst in by_det.items():
        mdists = [x['mean_dist'] for x in lst]
        nm = [x['num_matches'] for x in lst]
        summary.append({
            'detector': det,
            'pairs': len(lst),
            'avg_mean_dist': mean(mdists),
            'median_mean_dist': median(mdists),
            'std_mean_dist': pstdev(mdists) if len(mdists) > 1 else 0.0,
            'total_matches': sum(nm),
            'matches_per_pair': mean(nm)
        })
    return summary

def plot_bar(rows):
    # bar per pair
    labels = [f"{r['detector']}-{r['pair']}" for r in rows]
    values = [r['mean_dist'] for r in rows]
    plt.figure(figsize=(8,4))
    plt.bar(labels, values, color='steelblue')
    plt.ylabel('Mean Hamming Distance')
    plt.title('Mean Match Distance per Image Pair')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('results/histogram.png')
    plt.close()

if __name__ == "__main__":
    rows = read_matches('results/matches.txt')
    plot_bar(rows)
    for s in summarize(rows):
        print(s)