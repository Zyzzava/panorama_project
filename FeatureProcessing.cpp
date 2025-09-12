#include "FeatureProcessing.hpp"
#include <sstream>
#include <numeric>
#include <opencv2/stitching/detail/blenders.hpp>

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
        cv::Mat gray;
        if (images[i].channels() == 1)
            gray = images[i];
        else
            cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);

        cv::TickMeter tm;
        tm.start();
        det->detectAndCompute(gray, cv::noArray(), fs.kps[i], fs.descs[i]);
        tm.stop();
        fs.perImageDetectMs[i] = tm.getTimeMilli();
    }
    total.stop();
    fs.totalDetectMs = total.getTimeMilli();
    return fs;
}

static void projectCorners(const cv::Mat &img, const cv::Mat &H, std::vector<cv::Point2f> &out)
{
    std::vector<cv::Point2f> c = {
        {0, 0}, {(float)img.cols, 0}, {(float)img.cols, (float)img.rows}, {0, (float)img.rows}};
    cv::perspectiveTransform(c, out, H);
}

cv::Mat stitchTriple(const std::vector<cv::Mat> &imgs,
                     const cv::Mat &H_12,
                     const cv::Mat &H_23,
                     bool feather)
{
    if (imgs.size() < 3 || H_12.empty() || H_23.empty())
        return cv::Mat();

    // Reference: image 2 (center)-  otherwise distortion
    const cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat H1 = H_12;       // img1 -> img2
    const cv::Mat H2 = I;          // img2 -> img2
    const cv::Mat H3 = H_23.inv(); // img3 -> img2

    // finding the bounds in the panorama
    std::vector<cv::Point2f> c1, c2, c3, all;
    projectCorners(imgs[0], H1, c1);
    projectCorners(imgs[1], H2, c2);
    projectCorners(imgs[2], H3, c3);
    all.reserve(12);
    all.insert(all.end(), c1.begin(), c1.end());
    all.insert(all.end(), c2.begin(), c2.end());
    all.insert(all.end(), c3.begin(), c3.end());

    float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
    for (const auto &p : all)
    {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    }

    cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, -minX, 0, 1, -minY, 0, 0, 1);
    int W = std::max(1, (int)std::ceil(maxX - minX));
    int Hh = std::max(1, (int)std::ceil(maxY - minY));

    // Please don't explode
    const int MAX_SIDE = 6000;
    const double MAX_PIXELS = 16e6;
    double s_side = (double)MAX_SIDE / std::max(W, Hh);
    double s_area = std::sqrt(MAX_PIXELS / std::max(1.0, (double)W * (double)Hh));
    double s = std::min(1.0, std::min(s_side, s_area));

    cv::Mat S = (cv::Mat_<double>(3, 3) << s, 0, 0, 0, s, 0, 0, 0, 1);
    cv::Mat ST = S * T;
    int Ws = std::max(1, (int)std::ceil(W * s));
    int Hs = std::max(1, (int)std::ceil(Hh * s));

    // Bleding,  NO means simple overlay, FEATHER = soft blend
    int mode = feather ? cv::detail::Blender::FEATHER : cv::detail::Blender::NO;
    cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(mode, false);
    if (feather)
    {
        if (auto *fb = dynamic_cast<cv::detail::FeatherBlender *>(blender.get()))
            fb->setSharpness(0.02f);
    }
    blender->prepare(cv::Rect(0, 0, Ws, Hs));

    auto toBGR = [](const cv::Mat &src) -> cv::Mat
    {
        if (src.channels() == 3)
            return src;
        cv::Mat dst;
        if (src.channels() == 1)
            cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
        else
            cv::cvtColor(src, dst, cv::COLOR_BGRA2BGR);
        return dst;
    };

    auto warpAndFeed = [&](const cv::Mat &src, const cv::Mat &Hsrc)
    {
        cv::Mat src3 = toBGR(src);

        cv::Mat warped;
        cv::warpPerspective(src3, warped, ST * Hsrc, cv::Size(Ws, Hs),
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat mask(src.size(), CV_8U, cv::Scalar(255)), warpedMask;
        cv::warpPerspective(mask, warpedMask, ST * Hsrc, cv::Size(Ws, Hs),
                            cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

        cv::Mat warped_s;
        warped.convertTo(warped_s, CV_16SC3);
        blender->feed(warped_s, warpedMask, cv::Point(0, 0));
    };

    // For NO blending, last fed image wins in overlaps
    warpAndFeed(imgs[1], H2); // center
    warpAndFeed(imgs[0], H1); // left
    warpAndFeed(imgs[2], H3); // right

    cv::Mat result_s, result_mask;
    blender->blend(result_s, result_mask);

    cv::Mat result;
    result_s.convertTo(result, imgs[0].type());
    return result;
}