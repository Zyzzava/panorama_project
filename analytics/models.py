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

@dataclass
class HomographyResult:
    dataset: str         
    detector: str        
    img_i: int
    img_j: int
    threshold: float   
    num_inliers: int
    time_ms: float