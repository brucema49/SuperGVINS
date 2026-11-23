#pragma once
#include "frontend_policy.h"
#include "ort_include/onnxruntime_cxx_api.h"
#include "camodocal/camera_models/CameraFactory.h"
#include <opencv2/core.hpp>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace supergvins {
struct Feature {
    cv::Point2f model, pixel, normalized, velocity;
    std::array<float,256> descriptor;
    int id = -1;
    int age = 1;
};
struct FrameStats {
    size_t extractor_calls = 0, raw = 0, candidates = 0, previous = 0, previous_candidates = 0;
    size_t raw_matches = 0, matches = 0, ransac_inliers_before_cap = 0, inliers = 0, retained = 0;
    double extract_ms = 0, match_ms = 0, total_ms = 0;
};
class LearnedFrontend {
public:
    LearnedFrontend(const std::string& extractor, const std::string& matcher,
                    const std::string& camera, const std::string& device,
                    float threshold, double ransac_threshold, int min_distance);
    std::vector<Feature> process(const cv::Mat& gray, double timestamp, FrameStats& stats);
    void reset();
private:
    std::vector<Feature> extract(const cv::Mat& gray, FrameStats& stats);
    std::vector<Match> match(const std::vector<Feature>& current, int width, int height, FrameStats& stats);
    cv::Point2f normalized(const cv::Point2f& pixel) const;
    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> extractor_, matcher_;
    camodocal::CameraPtr camera_;
    std::vector<Feature> previous_candidates_;
    double previous_time_ = 0;
    int next_id_ = 0;
    float threshold_;
    double ransac_threshold_;
    int min_distance_;
};
}
