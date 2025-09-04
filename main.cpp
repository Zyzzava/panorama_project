#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>
#include <string>

struct FeatSet
{
    std::string name;
    std::vector<std::vector<cv::KeyPoint>> kps;
    std::vector<cv::Mat> descs;
    std::vector<double> perImageDetectMs;
    double totalDetectMs = 0.0;
};

static FeatSet runDetector(const std::string &name,
                           const cv::Ptr<cv::Feature2D> &det,
                           const std::vector<cv::Mat> &images)
{
    FeatSet fs;
    fs.name = name;
    fs.kps.resize(images.size());
    fs.descs.resize(images.size());
    fs.perImageDetectMs.resize(images.size(), 0.0);

    cv::TickMeter total;
    total.start();
    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::TickMeter tm;
        tm.start();
        det->detectAndCompute(images[i], cv::noArray(), fs.kps[i], fs.descs[i]);
        tm.stop();
        fs.perImageDetectMs[i] = tm.getTimeMilli();
    }
    total.stop();
    fs.totalDetectMs = total.getTimeMilli();
    return fs;
}

static void matchAndReport(const FeatSet &fs, std::ofstream &outputFile)
{
    if (fs.descs.size() < 2)
        return;
    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // crossCheck=true
    for (size_t i = 0; i < fs.descs.size(); ++i)
    {
        for (size_t j = i + 1; j < fs.descs.size(); ++j)
        {
            if (fs.descs[i].empty() || fs.descs[j].empty())
                continue;

            std::vector<cv::DMatch> matches;
            cv::TickMeter tm;
            tm.start();
            matcher.match(fs.descs[i], fs.descs[j], matches);
            tm.stop();

            double meanDist = 0.0;
            for (auto &m : matches)
                meanDist += m.distance;
            if (!matches.empty())
                meanDist /= matches.size();

            double matchTimeMs = tm.getTimeMilli();

            std::ostringstream distStream;
            for (size_t k = 0; k < matches.size(); ++k)
            {
                if (k)
                    distStream << ';';
                distStream << matches[k].distance;
            }

            // CSV: type,detector,img_i,img_j,num_matches,mean_dist,time_ms,distances
            outputFile << "match," << fs.name << ","
                       << (i + 1) << "," << (j + 1) << ","
                       << matches.size() << ","
                       << meanDist << ","
                       << matchTimeMs << ","
                       << distStream.str() << "\n";
        }
    }
}

int main()
{
    // Results exists, can't get cpp to make it ...:/

    std::vector<std::string> datasetFolders = {
        "images/pan1",
        "images/pan2",
        "images/pan3"};

    std::vector<std::pair<std::string, cv::Ptr<cv::Feature2D>>> detectors = {
        {"ORB", cv::ORB::create(1000)},
        {"AKAZE", cv::AKAZE::create()}};

    struct Summary
    {
        std::string dataset;
        std::string detector;
        size_t totalKps = 0;
        double avgKps = 0.0;
        double totalDetectMs = 0.0;
        double perImageDetectAvgMs = 0.0;
    };
    std::vector<Summary> summaries;

    for (size_t dsi = 0; dsi < datasetFolders.size(); ++dsi)
    {
        const std::string &folder = datasetFolders[dsi];

        std::vector<cv::String> found;
        cv::glob(folder + "/*.png", found, false);
        if (found.size() < 3)
            std::cerr << "Warning: folder " << folder << " has " << found.size() << " png images (>=3 expected)\n";

        std::vector<cv::Mat> images;
        images.reserve(found.size());
        for (auto &p : found)
        {
            cv::Mat g = cv::imread(p, cv::IMREAD_GRAYSCALE);
            if (g.empty())
            {
                std::cerr << "Failed to load: " << p << "\n";
                return 1;
            }
            images.push_back(g);
        }
        if (images.empty())
        {
            std::cerr << "No images loaded for folder " << folder << "\n";
            continue;
        }

        std::cout << "=== Dataset " << (dsi + 1) << " (" << folder << ") ===\n";

        std::string outPath = "results/results" + std::to_string(dsi + 1) + ".txt";
        std::ofstream outputFile(outPath);
        if (!outputFile.is_open())
        {
            std::cerr << "Failed to open " << outPath << " for writing\n";
            continue;
        }

        // Header so that histogram.py understands
        outputFile << "type,detector,img_i,img_j,num_matches,mean_dist,time_ms,distances\n";

        for (auto &detPair : detectors)
        {
            const std::string &detName = detPair.first;
            auto detPtr = detPair.second;

            FeatSet fs = runDetector(detName, detPtr, images);

            size_t totalKps = 0;
            for (auto &vec : fs.kps)
                totalKps += vec.size();
            double avgKps = fs.kps.empty() ? 0.0 : static_cast<double>(totalKps) / fs.kps.size();

            // Detection rows: mean_dist empty, time_ms = per-image detect time, distances tag
            for (size_t i = 0; i < fs.kps.size(); ++i)
            {
                outputFile << "detect," << detName << "," << (i + 1) << ",,"
                           << fs.kps[i].size() << "," // num_matches = keypoint count
                           << "" << ","               // mean_dist empty
                           << fs.perImageDetectMs[i] << ","
                           << "kps_per_image\n";
            }

            // Matching rows
            matchAndReport(fs, outputFile);

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