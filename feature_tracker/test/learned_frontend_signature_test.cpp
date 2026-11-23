#include "learned_frontend.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string constructorError(const std::string& extractor,
                             const std::string& matcher,
                             const std::string& camera)
{
    try
    {
        supergvins::LearnedFrontend frontend(
            extractor, matcher, camera, "cpu", 0.5f, 3.0, 20);
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    return std::string();
}

bool contains(const std::string& value, const std::string& fragment)
{
    return value.find(fragment) != std::string::npos;
}

bool validFeatures(const std::vector<supergvins::Feature>& features,
                   int width,
                   int height,
                   double scale,
                   const camodocal::CameraPtr& camera)
{
    for (const supergvins::Feature& feature : features)
    {
        if (feature.id < 0 || feature.pixel.x < 1 || feature.pixel.y < 1 ||
            feature.pixel.x >= width - 1 || feature.pixel.y >= height - 1)
            return false;
        const float expected_x = (feature.model.x + 0.5f) / static_cast<float>(scale) - 0.5f;
        const float expected_y = (feature.model.y + 0.5f) / static_cast<float>(scale) - 0.5f;
        if (std::abs(feature.pixel.x - expected_x) > 1e-4f ||
            std::abs(feature.pixel.y - expected_y) > 1e-4f)
            return false;

        Eigen::Vector2d round_trip;
        camera->spaceToPlane(
            Eigen::Vector3d(feature.normalized.x, feature.normalized.y, 1.0), round_trip);
        if ((round_trip - Eigen::Vector2d(feature.pixel.x, feature.pixel.y)).norm() > 1e-2)
            return false;
        for (float value : feature.descriptor)
            if (!std::isfinite(value))
                return false;
    }
    return true;
}

void addLowFeatureShapes(cv::Mat& image, int offset)
{
    cv::rectangle(image, cv::Point(38 + offset, 42), cv::Point(62 + offset, 66), cv::Scalar(255), -1);
    cv::rectangle(image, cv::Point(238 + offset, 158), cv::Point(262 + offset, 182), cv::Scalar(255), -1);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: learned_frontend_signature_test superpoint.onnx "
                     "lightglue.onnx camera_config.yaml\n";
        return 2;
    }

    const std::string extractor = argv[1];
    const std::string matcher = argv[2];
    const std::string camera = argv[3];

    const std::string valid_error = constructorError(extractor, matcher, camera);
    if (!valid_error.empty())
    {
        std::cerr << "valid model pair rejected: " << valid_error << '\n';
        return 1;
    }

    const std::string swapped_error = constructorError(matcher, extractor, camera);
    if (!contains(swapped_error, "SuperPoint extractor model signature mismatch"))
    {
        std::cerr << "swapped model pair did not report the extractor signature mismatch clearly: "
                  << swapped_error << '\n';
        return 1;
    }

    const std::string missing_extractor_error =
        constructorError(extractor + ".missing", matcher, camera);
    if (!contains(missing_extractor_error, "failed to load SuperPoint extractor model"))
    {
        std::cerr << "missing extractor did not identify its model role: "
                  << missing_extractor_error << '\n';
        return 1;
    }

    const std::string missing_matcher_error =
        constructorError(extractor, matcher + ".missing", camera);
    if (!contains(missing_matcher_error, "failed to load LightGlue matcher model"))
    {
        std::cerr << "missing matcher did not identify its model role: "
                  << missing_matcher_error << '\n';
        return 1;
    }

    supergvins::LearnedFrontend frontend(
        extractor, matcher, camera, "cpu", 0.5f, 3.0, 20);
    supergvins::FrameStats stats;
    cv::Mat blank = cv::Mat::zeros(240, 320, CV_8UC1);
    const std::vector<supergvins::Feature> blank_features = frontend.process(blank, 1.0, stats);
    if (!blank_features.empty() || stats.raw != 0 || stats.candidates != 0 || stats.retained != 0)
    {
        std::cerr << "blank image should produce no keypoints or tracks; raw=" << stats.raw
                  << " candidates=" << stats.candidates << " retained=" << stats.retained << '\n';
        return 1;
    }

    const double scale = 512.0 / 320.0;
    cv::Mat low_feature = cv::Mat::zeros(240, 320, CV_8UC1);
    addLowFeatureShapes(low_feature, 0);
    const std::vector<supergvins::Feature> first = frontend.process(low_feature, 1.05, stats);
    const size_t first_candidate_count = stats.candidates;
    if (stats.candidates == 0 || stats.candidates >= 128 || first.empty() || first.size() >= 128)
    {
        std::cerr << "low-feature non-square image did not exercise padded matching; candidates="
                  << stats.candidates << " tracks=" << first.size() << '\n';
        return 1;
    }
    if (stats.previous_candidates != 0 || stats.previous != 0 || stats.retained != first.size())
    {
        std::cerr << "first image reported an unexpected previous cache or active track count\n";
        return 1;
    }
    camodocal::CameraPtr camera_model =
        camodocal::CameraFactory::instance()->generateCameraFromYamlFile(camera);
    if (!validFeatures(first, low_feature.cols, low_feature.rows, scale, camera_model))
    {
        std::cerr << "low-feature outputs contained padding or failed coordinate round-trip\n";
        return 1;
    }

    cv::Mat low_feature_next = cv::Mat::zeros(240, 320, CV_8UC1);
    addLowFeatureShapes(low_feature_next, 1);
    const std::vector<supergvins::Feature> second = frontend.process(low_feature_next, 1.10, stats);
    if (stats.previous_candidates != first_candidate_count || stats.previous > 200 ||
        stats.candidates == 0 || stats.candidates >= 128 ||
        stats.matches > std::min(stats.previous_candidates, stats.candidates) ||
        stats.matches > 512 || stats.ransac_inliers_before_cap > stats.matches ||
        stats.inliers != std::min<size_t>(stats.ransac_inliers_before_cap, 200) ||
        second.size() > 200 || stats.retained != second.size())
    {
        std::cerr << "padded LightGlue inputs escaped the valid candidate range; previous="
                  << stats.previous << " previous_candidates=" << stats.previous_candidates
                  << " candidates=" << stats.candidates << " matches=" << stats.matches
                  << " pre_cap_inliers=" << stats.ransac_inliers_before_cap
                  << " retained_matches=" << stats.inliers << " tracks=" << second.size() << '\n';
        return 1;
    }
    for (size_t i = 0; i < second.size(); ++i)
        for (size_t j = i + 1; j < second.size(); ++j)
            if (second[i].id == second[j].id)
            {
                std::cerr << "duplicate active track IDs after padded matching\n";
                return 1;
            }
    if (!validFeatures(second, low_feature_next.cols, low_feature_next.rows, scale, camera_model))
    {
        std::cerr << "matched low-feature outputs contained padding or failed coordinate round-trip\n";
        return 1;
    }

    std::cout << "learned frontend signature and CPU pipeline checks passed\n";
    return 0;
}
