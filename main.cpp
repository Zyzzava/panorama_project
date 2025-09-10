#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <vector>
#include <string>
#include "FeatureProcessing.hpp"

struct Summary
{
    std::string dataset;
    std::string detector;
    size_t totalKps = 0;
    double avgKps = 0.0;
    double totalDetectMs = 0.0;
    double perImageDetectAvgMs = 0.0;
};

int main()
{
    std::vector<std::string> datasetFolders = {
        "images/pan1",
        "images/pan2",
        "images/pan3"};

    std::vector<std::pair<std::string, cv::Ptr<cv::Feature2D>>> detectors = {
        {"ORB", cv::ORB::create(100000)},
        {"AKAZE", cv::AKAZE::create()}};

    std::vector<Summary> summaries;

    for (size_t dsi = 0; dsi < datasetFolders.size(); ++dsi)
    {
        const std::string &folder = datasetFolders[dsi];
        std::vector<cv::String> found;
        cv::glob(folder + "/*.png", found, false);

        std::vector<cv::Mat> images;
        images.reserve(found.size());
        for (auto &p : found)
        {
            cv::Mat g = cv::imread(p, cv::IMREAD_COLOR);
            images.push_back(g);
        }

        std::cout << "=== Dataset " << (dsi + 1) << " (" << folder << ") ===\n";

        std::string outPath = "results/results" + std::to_string(dsi + 1) + ".txt";
        std::ofstream outputFile(outPath);
        outputFile << "type,detector,img_i,img_j,num_matches,mean_dist,time_ms,distances\n";

        std::string homographyPath = "results/homography_results" + std::to_string(dsi + 1) + ".txt";
        std::ofstream homographyFile(homographyPath);
        homographyFile << "type,detector,img_i,img_j,num_inliers,time_ms,threshold\n";

        for (auto &detPair : detectors)
        {
            const std::string &detName = detPair.first;
            auto detPtr = detPair.second;

            FeatSet fs = runDetector(detName, detPtr, images);

            size_t totalKps = 0;
            for (auto &vec : fs.kps)
                totalKps += vec.size();
            double avgKps = fs.kps.empty() ? 0.0 : static_cast<double>(totalKps) / fs.kps.size();

            for (size_t i = 0; i < fs.kps.size(); ++i)
            {
                outputFile << "detect," << detName << "," << (i + 1) << ",,"
                           << fs.kps[i].size() << ","
                           << "" << ","
                           << fs.perImageDetectMs[i] << ","
                           << "kps_per_image\n";
            }

            matchAndReport(fs, outputFile);

            homographyExperiments(fs, images, homographyFile, detName);

            std::vector<double> ransacThresholds = {1.0, 5.0, 15.0};
            for (double thr : ransacThresholds)
            {
                cv::Mat H_12 = computeHomographyOneWay(fs.kps[0], fs.kps[1], fs.descs[0], fs.descs[1], thr);
                cv::Mat H_23 = computeHomographyOneWay(fs.kps[1], fs.kps[2], fs.descs[1], fs.descs[2], thr);

                cv::Mat pano_over = stitchTriple(images, H_12, H_23, false);
                cv::Mat pano_feather = stitchTriple(images, H_12, H_23, true);

                if (!pano_over.empty())
                    cv::imwrite("results/panorama_ds" + std::to_string(dsi + 1) +
                                    "_" + detName + "_thr" + std::to_string(thr) + "_over.png",
                                pano_over);
                if (!pano_feather.empty())
                    cv::imwrite("results/panorama_ds" + std::to_string(dsi + 1) +
                                    "_" + detName + "_thr" + std::to_string(thr) + "_feather.png",
                                pano_feather);
            }

            /*
            // --- Simple visualization of matches for first 3 images ---
            if (fs.descs.size() >= 3)
            {
                auto showPair = [&](int a, int b)
                {
                    if (fs.descs[a].empty() || fs.descs[b].empty())
                        return;
                    int normType = (fs.descs[a].type() == CV_8U) ? cv::NORM_HAMMING : cv::NORM_L2;
                    cv::BFMatcher matcher(normType, true);
                    std::vector<cv::DMatch> matches;
                    matcher.match(fs.descs[a], fs.descs[b], matches);
                    if (matches.empty())
                        return;
                    std::sort(matches.begin(), matches.end(),
                              [](const cv::DMatch &m1, const cv::DMatch &m2)
                              { return m1.distance < m2.distance; });
                    if (matches.size() > 50)
                        matches.resize(50);
                    cv::Mat vis;
                    cv::drawMatches(images[a], fs.kps[a],
                                    images[b], fs.kps[b],
                                    matches, vis);
                    std::string win = "DS" + std::to_string(dsi + 1) + " " + detName +
                                      " " + std::to_string(a + 1) + "-" + std::to_string(b + 1);
                    cv::imshow(win, vis);
                };
                showPair(0, 1);
                showPair(1, 2);
                cv::waitKey(0); // wait for key press before continuing
                cv::destroyAllWindows();
            }
            // --- end visualization ---
            */

            double perImgAvgMs = fs.kps.empty() ? 0.0 : fs.totalDetectMs / fs.kps.size();
            summaries.push_back({folder, detName, totalKps, avgKps, fs.totalDetectMs, perImgAvgMs});

            std::cout << detName
                      << " total_kps=" << totalKps
                      << " avg_kps=" << avgKps
                      << " total_detect_ms=" << fs.totalDetectMs
                      << " per_image_detect_ms=" << perImgAvgMs << "\n";
        }
    }

    std::cout << "=== Overview (per dataset & detector) ===\n";
    for (auto &s : summaries)
    {
        std::cout << s.dataset << "," << s.detector
                  << ",total_kps=" << s.totalKps
                  << ",avg_kps=" << s.avgKps
                  << ",total_detect_ms=" << s.totalDetectMs
                  << ",per_image_detect_ms=" << s.perImageDetectAvgMs << "\n";
    }
    std::cout << "Details in results/resultsX.txt files.\n";
    return 0;
}
