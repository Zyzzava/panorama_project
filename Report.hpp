#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "FeatureProcessing.hpp"

// Used to store homography results
struct HomogEntry
{
    size_t i; // 0-based img index
    size_t j; // 0-based img index
    double thr;
    cv::Mat H;
    int inliers;
    double timeMs;
};

/*
Run the matching and homography experiments and write results to outputFile.
*/
void matchAndHomographyReport(const FeatSet &fs,
                              std::ofstream &kps_Match_File,
                              std::ofstream &homography_File,
                              const std::string &detName,
                              std::vector<HomogEntry> *outHomogs = nullptr);