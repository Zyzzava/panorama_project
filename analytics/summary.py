from collections import defaultdict
from statistics import mean, median, pstdev
from typing import List, Dict, Tuple
from .models import MatchResult, MatchSummary

def summarize_matches(matches: List[MatchResult]) -> List[MatchSummary]:
    grouped: Dict[Tuple[str,str], List[MatchResult]] = defaultdict(list)
    for m in matches:
        grouped[(m.dataset, m.detector)].append(m)
    out: List[MatchSummary] = []
    for (ds, det), lst in grouped.items():
        mdists = [x.mean_dist for x in lst]
        nm = [x.num_matches for x in lst]
        mt = [x.time_ms for x in lst]
        out.append(MatchSummary(
            dataset=ds,
            detector=det,
            pairs=len(lst),
            avg_mean_dist=mean(mdists),
            median_mean_dist=median(mdists),
            std_mean_dist=pstdev(mdists) if len(mdists) > 1 else 0.0,
            total_matches=sum(nm),
            avg_matches_per_pair=mean(nm),
            avg_match_time_ms=mean(mt),
            total_match_time_ms=sum(mt)
        ))
    return out