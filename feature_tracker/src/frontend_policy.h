#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace supergvins {
constexpr size_t kMaxCandidateCount = 512;
constexpr size_t kMaxActiveTrackCount = 200;

struct Match { int previous; int current; float score; };
struct MatchedTrackState { int id; int age; float velocity_x; float velocity_y; };

inline bool matchQualityBefore(const Match& a, const Match& b)
{
    if (a.score != b.score) return a.score > b.score;
    if (a.previous != b.previous) return a.previous < b.previous;
    return a.current < b.current;
}

inline std::vector<size_t> selectKeypoints(const std::vector<float>& scores, size_t limit)
{
    for (float score : scores)
        if (!std::isfinite(score)) throw std::runtime_error("non-finite SuperPoint score");
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    const auto count = std::min(limit, scores.size());
    if (scores.size() > limit) {
        std::partial_sort(order.begin(), order.begin() + count, order.end(),
            [&scores](size_t a, size_t b) { return scores[a] == scores[b] ? a < b : scores[a] > scores[b]; });
        order.resize(count);
    }
    return order;
}
inline std::vector<Match> selectMatches(const std::vector<Match>& candidates, size_t prev_count,
                                        size_t cur_count, float threshold, size_t limit)
{
    if (limit == 0) return {};
    std::vector<Match> ordered;
    ordered.reserve(candidates.size());
    for (const auto& m : candidates) {
        if (!std::isfinite(m.score)) throw std::runtime_error("non-finite LightGlue score");
        if (m.previous < 0 || m.current < 0 || static_cast<size_t>(m.previous) >= prev_count ||
            static_cast<size_t>(m.current) >= cur_count || m.score <= threshold) continue;
        ordered.push_back(m);
    }
    std::sort(ordered.begin(), ordered.end(), matchQualityBefore);
    std::vector<Match> result;
    std::vector<bool> used_prev(prev_count), used_cur(cur_count);
    result.reserve(std::min(limit, ordered.size()));
    for (const auto& m : ordered) {
        if (used_prev[m.previous] || used_cur[m.current]) continue;
        used_prev[m.previous] = used_cur[m.current] = true;
        result.push_back(m);
        if (result.size() >= limit) break;
    }
    return result;
}

inline void capMatchesInPlace(std::vector<Match>& candidates, size_t limit)
{
    if (candidates.size() > limit) candidates.resize(limit);
}

inline void capMatchesByTrackAge(std::vector<Match>& candidates, size_t limit,
                                 const std::vector<int>& previous_track_ages)
{
    // Call only after score filtering, uniqueness checks and geometric verification.
    // Cached candidates without an assigned track ID have age zero for this selection.
    std::sort(candidates.begin(), candidates.end(), [&previous_track_ages](const Match& a, const Match& b) {
        const int age_a = previous_track_ages.at(a.previous);
        const int age_b = previous_track_ages.at(b.previous);
        if (age_a != age_b) return age_a > age_b;
        return matchQualityBefore(a, b);
    });
    capMatchesInPlace(candidates, limit);
}

inline MatchedTrackState advanceMatchedTrack(int previous_id, int previous_age,
                                              float previous_x, float previous_y,
                                              float current_x, float current_y,
                                              double delta_time, int& next_id)
{
    if (previous_age < 1 || !std::isfinite(delta_time) || delta_time <= 0.0 ||
        !std::isfinite(previous_x) || !std::isfinite(previous_y) ||
        !std::isfinite(current_x) || !std::isfinite(current_y))
        throw std::runtime_error("invalid matched track state or timestamp interval");
    int id = previous_id;
    if (id < 0) {
        if (next_id < 0 || next_id >= (1 << 24))
            throw std::runtime_error("feature id exhausted float32 precision; coordinated restart required");
        id = next_id++;
    }
    return {id, previous_age + 1,
            static_cast<float>((current_x - previous_x) / delta_time),
            static_cast<float>((current_y - previous_y) / delta_time)};
}
}
