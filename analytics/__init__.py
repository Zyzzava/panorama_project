from .reader import load_all, load_homographies
from .plotting import (
    plot_mean_dist_bar,
    plot_distance_histograms,
    plot_keypoints_bar,
    plot_match_time_bar,
    plot_homography_inliers,
    plot_homography_time,
    plot_homography_summary_grid,
)

def main():
    matches, detect_summary = load_all()
    homos = load_homographies()
    plot_mean_dist_bar(matches)
    plot_match_time_bar(matches)
    plot_distance_histograms(matches)
    plot_keypoints_bar(detect_summary)
    plot_homography_inliers(homos)
    plot_homography_time(homos)
    plot_homography_summary_grid(homos)

__all__ = ["main"]