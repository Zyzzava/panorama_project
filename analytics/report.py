from typing import Dict, Tuple, List
from .models import DetectResult, MatchSummary

def print_overview(detects: Dict[Tuple[str,str], DetectResult], summaries: List[MatchSummary]):
    print("=== Overview (per dataset & detector) ===")
    ms_index = {(m.dataset, m.detector): m for m in summaries}
    for key, detect in sorted(detects.items()):
        ms = ms_index.get(key)
        print(
            f"D{detect.dataset}-{detect.detector}: "
            f"total_kp={detect.total_kp}, detect_time_ms={detect.detect_time_ms:.2f}, "
            f"pairs={ms.pairs if ms else '-'}, "
            f"avg_match_time_ms={ms.avg_match_time_ms if ms else '-'}, "
            f"avg_mean_dist={ms.avg_mean_dist if ms else '-'}"
        )