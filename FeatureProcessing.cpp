#include "FeatureProcessing.hpp"
#include <sstream>
#include <numeric>

FeatSet runDetector(const std::string &name,
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

void matchAndReport(const FeatSet &fs, std::ofstream &outputFile)
{
    if (fs.descs.size() < 2)
        return;

    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // crossCheck = true

    // Only adjacent pairs: 1-2, 2-3, ...
    for (size_t i = 0; i + 1 < fs.descs.size(); ++i)
    {
        size_t j = i + 1;

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
            distStream << matches[k].distance;
            if (k + 1 < matches.size())
                distStream << ";";
        }

        outputFile << "match,"
                   << fs.name << ","
                   << (i + 1) << ","
                   << (j + 1) << ","
                   << matches.size() << ","
                   << meanDist << ","
                   << matchTimeMs << ","
                   << distStream.str() << "\n";
    }
}

void homographyExperiments(const FeatSet &fs,
                           const std::vector<cv::Mat> &images,
                           std::ofstream &outputFile,
                           const std::string &detName)
{
    if (fs.descs.size() < 2)
        return;

    // Small set of reprojection thresholds (pixels)
    const std::vector<double> thresholds{1.0, 2.0, 3.0};

    for (size_t i = 0; i + 1 < fs.descs.size(); ++i)
    {
        size_t j = i + 1;
        if (fs.descs[i].empty() || fs.descs[j].empty())
            continue;

        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(fs.descs[i], fs.descs[j], matches);
        if (matches.size() < 4)
            continue;

        std::vector<cv::Point2f> ptsA, ptsB;
        ptsA.reserve(matches.size());
        ptsB.reserve(matches.size());
        for (auto &m : matches)
        {
            ptsA.push_back(fs.kps[i][m.queryIdx].pt);
            ptsB.push_back(fs.kps[j][m.trainIdx].pt);
        }

        for (double thr : thresholds)
        {
            std::vector<unsigned char> inlierMask;
            cv::TickMeter tm;
            tm.start();
            cv::Mat H = cv::findHomography(ptsA, ptsB, cv::RANSAC, thr, inlierMask);
            tm.stop();

            if (H.empty())
                continue;

            int inliers = std::accumulate(inlierMask.begin(), inlierMask.end(), 0);
            double inlierRatio = matches.empty() ? 0.0 : static_cast<double>(inliers) / matches.size();

            outputFile << "homography,"
                       << detName << ","
                       << (i + 1) << ","
                       << (j + 1) << ","
                       << inliers << ","
                       << inlierRatio << ","
                       << tm.getTimeMilli() << ","
                       << "thr=" << thr << "\n";
        }
    }
}