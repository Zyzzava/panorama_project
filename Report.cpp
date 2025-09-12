#include "Report.hpp"

#include <sstream>
#include <numeric>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>

void matchAndHomographyReport(const FeatSet &fs,
                              std::ofstream &kps_Match_File,
                              std::ofstream &homography_File,
                              const std::string &detName,
                              std::vector<HomogEntry> *outHomogs)
{
    if (fs.descs.size() < 2)
        return;

    const std::vector<double> thresholds{1.0, 5.0, 15.0};

    // Reuse one matcher instance
    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // crossCheck = true

    // Only adjacent pairs: 1-2, 2-3, ...
    for (size_t i = 0; i + 1 < fs.descs.size(); ++i)
    {
        size_t j = i + 1;
        if (fs.descs[i].empty() || fs.descs[j].empty())
            continue;

        // Match once
        std::vector<cv::DMatch> matches;
        cv::TickMeter tmMatch;
        tmMatch.start();
        matcher.match(fs.descs[i], fs.descs[j], matches);
        tmMatch.stop();

        // Match stats
        double meanDist = 0.0;
        for (const auto &m : matches)
            meanDist += m.distance;
        if (!matches.empty())
            meanDist /= matches.size();

        std::ostringstream distStream;
        for (size_t k = 0; k < matches.size(); ++k)
        {
            distStream << matches[k].distance;
            if (k + 1 < matches.size())
                distStream << ";";
        }

        kps_Match_File << "match,"
                       << fs.name << ","
                       << (i + 1) << ","
                       << (j + 1) << ","
                       << matches.size() << ","
                       << meanDist << ","
                       << tmMatch.getTimeMilli() << ","
                       << distStream.str() << "\n";

        if (matches.size() < 4)
            continue;

        // Reuse matched indices to build point sets
        std::vector<cv::Point2f> ptsA, ptsB;
        ptsA.reserve(matches.size());
        ptsB.reserve(matches.size());
        for (const auto &m : matches)
        {
            ptsA.push_back(fs.kps[i][m.queryIdx].pt);
            ptsB.push_back(fs.kps[j][m.trainIdx].pt);
        }

        // Run the homography at different thresholds
        for (double thr : thresholds)
        {
            std::vector<unsigned char> inlierMask;
            cv::TickMeter tmH;
            tmH.start();
            cv::Mat H = cv::findHomography(ptsA, ptsB, cv::RANSAC, thr, inlierMask);
            tmH.stop();

            if (H.empty())
                continue;

            int inliers = std::accumulate(inlierMask.begin(), inlierMask.end(), 0);

            homography_File << "homography,"
                            << detName << ","
                            << (i + 1) << ","
                            << (j + 1) << ","
                            << inliers << ","
                            << tmH.getTimeMilli() << ","
                            << "thr=" << thr << "\n";

            if (outHomogs)
                outHomogs->push_back(HomogEntry{i, j, thr, H.clone(), inliers, tmH.getTimeMilli()});
        }
    }
}