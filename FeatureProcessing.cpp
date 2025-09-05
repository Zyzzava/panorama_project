#include "FeatureProcessing.hpp"
#include <sstream>

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