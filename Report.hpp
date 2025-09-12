#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "FeatureProcessing.hpp"

// Reports basic matching stats between adjacent images
void matchAndReport(const FeatSet &fs, std::ofstream &outputFile);

// Runs homography estimation experiments and reports inliers/time
void homographyExperiments(const FeatSet &fs,
                           const std::vector<cv::Mat> &images,
                           std::ofstream &outputFile,
                           const std::string &detName);