from .reader import load_all
from .summary import summarize_matches
from .plotting import (
    plot_mean_dist_bar,
    plot_distance_histograms,
    plot_keypoints_bar,
)
from .report import print_overview

def main():
    matches, detect_map = load_all()
    summaries = summarize_matches(matches)
    plot_mean_dist_bar(matches)
    plot_distance_histograms(matches)
    plot_keypoints_bar(detect_map)
    print_overview(detect_map, summaries)

__all__ = ["main"]