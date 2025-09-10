#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <fstream>

struct FeatSet
{
    std::string name;
    std::vector<std::vector<cv::KeyPoint>> kps;
    std::vector<cv::Mat> descs;
    std::vector<double> perImageDetectMs;
    double totalDetectMs = 0.0;
};

// Detect keypoints/descriptors for all images with given detector
FeatSet runDetector(const std::string &name,
                    const cv::Ptr<cv::Feature2D> &det,
                    const std::vector<cv::Mat> &images);

// Match all image pairs and append CSV rows
void matchAndReport(const FeatSet &fs, std::ofstream &outputFile);

// Homography-based experiments and append CSV rows
void homographyExperiments(const FeatSet &fs,
                           const std::vector<cv::Mat> &images,
                           std::ofstream &outputFile,
                           const std::string &detName);

cv::Mat computeHomographyOneWay(const std::vector<cv::KeyPoint> &kpsA,
                                const std::vector<cv::KeyPoint> &kpsB,
                                const cv::Mat &descA,
                                const cv::Mat &descB,
                                double ransacThresh);

cv::Mat stitchTriple(const std::vector<cv::Mat> &imgs,
                     const cv::Mat &H_12,
                     const cv::Mat &H_23,
                     bool feather); // feather=true => averaging, false => simple over