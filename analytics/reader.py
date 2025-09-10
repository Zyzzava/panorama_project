import csv
from glob import glob
from typing import List, Dict, Tuple
from .models import MatchResult, DetectResult, HomographyResult

def read_results_file(path: str, dataset_tag: str) -> Tuple[List[MatchResult], Dict[Tuple[str,str], DetectResult]]:
    matches: List[MatchResult] = []
    detects: Dict[Tuple[str,str], DetectResult] = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            t = row['type']
            if t == 'match':
                distances = []
                if row.get('distances'):
                    distances = [float(x) for x in row['distances'].split(';') if x]
                matches.append(
                    MatchResult(
                        dataset=dataset_tag,
                        detector=row['detector'],
                        pair=f"{row['img_i']}-{row['img_j']}",
                        num_matches=int(row['num_matches']),
                        mean_dist=float(row['mean_dist']),
                        time_ms=float(row['time_ms']),
                        distances=distances
                    )
                )
            elif t == 'detect':
                detects[(dataset_tag, row['detector'])] = DetectResult(
                    dataset=dataset_tag,
                    detector=row['detector'],
                    total_kp=int(row['num_matches']) if row['num_matches'] else 0,
                    detect_time_ms=float(row['time_ms']) if row['time_ms'] else 0.0
                )
    return matches, detects

def load_all(pattern: str = "results/results*.txt"):
    all_matches: List[MatchResult] = []
    detect_map: Dict[Tuple[str,str], DetectResult] = {}
    for path in sorted(glob(pattern)):
        tag = path.split("results")[-1].split(".")[0] or "?"
        m, d = read_results_file(path, tag)
        all_matches.extend(m)
        detect_map.update(d)
    return all_matches, detect_map

def read_homography_file(path: str, dataset_tag: str) -> List[HomographyResult]:
    homos: List[HomographyResult] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") != "homography":
                continue
            thr_raw = row.get("threshold","")
            if "=" in thr_raw:
                thr_raw = thr_raw.split("=",1)[1]
            try:
                thr = float(thr_raw)
            except ValueError:
                continue
            homos.append(
                HomographyResult(
                    dataset=dataset_tag,
                    detector=row["detector"],
                    img_i=int(row["img_i"]),
                    img_j=int(row["img_j"]),
                    threshold=thr,
                    num_inliers=int(row["num_inliers"]),
                    time_ms=float(row["time_ms"]),
                )
            )
    return homos

def load_homographies(pattern: str = "results/homography_results*.txt") -> List[HomographyResult]:
    all_h: List[HomographyResult] = []
    for path in sorted(glob(pattern)):
        tag = path.split("homography_results")[-1].split(".")[0] or "?"
        all_h.extend(read_homography_file(path, tag))
    return all_h