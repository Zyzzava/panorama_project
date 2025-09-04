#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>

struct FeatSet
{
    std::string name;
    std::vector<std::vector<cv::KeyPoint>> kps;
    std::vector<cv::Mat> descs;
    double tDetectMs = 0.0;
};

static FeatSet runDetector(const std::string &name,
                           const cv::Ptr<cv::Feature2D> &det,
                           const std::vector<cv::Mat> &images)
{
    FeatSet fs;
    fs.name = name;
    fs.kps.resize(images.size());
    fs.descs.resize(images.size());

    cv::TickMeter tm; // Timer start
    tm.start();
    for (size_t i = 0; i < images.size(); ++i)
        det->detectAndCompute(images[i], cv::noArray(), fs.kps[i], fs.descs[i]);
    tm.stop();
    fs.tDetectMs = tm.getTimeMilli(); // Timer finish
    return fs;
}

static void matchAndReport(const FeatSet &fs, std::ofstream &outputFile)
{
    if (fs.descs.size() < 2)
        return;
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    for (size_t i = 0; i < fs.descs.size(); ++i)
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
            outputFile << "match," << fs.name << ","
                       << (i + 1) << "," << (j + 1) << ","
                       << matches.size() << ","
                       << meanDist << ","
                       << matchTimeMs << "\n";
        }
}

int main()
{
    std::vector<std::string> imgPaths = {
        "images/pan1/pan1.png",
        "images/pan1/pan2.png",
        "images/pan1/pan3.png"};
    std::vector<cv::Mat> images;
    images.reserve(imgPaths.size());
    for (const auto &p : imgPaths)
    {
        cv::Mat g = cv::imread(p, cv::IMREAD_GRAYSCALE);
        if (g.empty())
        {
            std::cerr << "Failed to load: " << p << "\n";
            return 1;
        }
        images.push_back(g);
    }

    std::vector<std::pair<std::string, cv::Ptr<cv::Feature2D>>> detectors = {
        {"ORB", cv::ORB::create(1000000)},
        {"AKAZE", cv::AKAZE::create()}};

    std::vector<FeatSet> results;
    std::ofstream outputFile("results/matches.txt");
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open results/matches.txt for writing.\n";
        return 1;
    }

    // CSV header
    outputFile << "type,detector,img_i,img_j,num_matches,mean_dist,time_ms\n";

    for (auto &d : detectors)
    {
        auto fs = runDetector(d.first, d.second, images);
        // write detect row
        outputFile << "detect," << fs.name << ",,,,," << fs.tDetectMs << "\n"; // these are empty placeholders for image/match fields
        results.push_back(std::move(fs));
    }

    for (auto &fs : results)
        matchAndReport(fs, outputFile);

    outputFile.close();
    return 0;
}