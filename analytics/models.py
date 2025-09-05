from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class MatchResult:
    dataset: str
    detector: str
    pair: str
    num_matches: int
    mean_dist: float
    time_ms: float
    distances: List[float]

@dataclass(frozen=True)
class DetectResult:
    dataset: str
    detector: str
    total_kp: int
    detect_time_ms: float

@dataclass(frozen=True)
class MatchSummary:
    dataset: str
    detector: str
    pairs: int
    avg_mean_dist: float
    median_mean_dist: float
    std_mean_dist: float
    total_matches: int
    avg_matches_per_pair: float
    avg_match_time_ms: float
    total_match_time_ms: float