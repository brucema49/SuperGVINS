#include "frontend_policy.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    std::vector<float> scores(513);
    for (size_t i = 0; i < scores.size(); ++i) scores[i] = static_cast<float>(i);
    scores[0] = scores[1] = 1000.0f;
    auto selected = supergvins::selectKeypoints(scores, 512);
    assert(selected.size() == 512);
    assert(selected[0] == 0 && selected[1] == 1);
    assert(selected.back() == 3);
    auto all = supergvins::selectKeypoints({0.3f, 0.7f}, 512);
    assert(all.size() == 2);
    auto valid = supergvins::selectMatches({{0,0,0.9f},{0,1,0.8f},{1,1,0.7f},{2,9,0.9f},{2,2,0.5f}}, 3, 3, 0.5f, 128);
    assert(valid.size() == 2);
    assert(valid[0].previous == 0 && valid[0].current == 0);
    assert(valid[1].previous == 1 && valid[1].current == 1);
    std::vector<supergvins::Match> many;
    for (int i=0; i<150; ++i) many.push_back({i,i,0.9f});
    assert(supergvins::selectMatches(many,150,150,0.5f,
                                     supergvins::kMaxCandidateCount).size() == 150);

    std::vector<supergvins::Match> over_budget;
    for (int i=0; i<257; ++i)
        over_budget.push_back({i,i,0.51f + i * 0.001f});
    const auto verified = supergvins::selectMatches(
        over_budget, 512, 512, 0.5f, supergvins::kMaxCandidateCount);
    assert(verified.size() == 257);
    assert(verified.front().previous == 256 && verified.back().previous == 0);
    std::vector<supergvins::Match> geometry_inliers;
    for (const auto& match : verified)
        if (match.previous % 5 != 0) geometry_inliers.push_back(match);
    assert(geometry_inliers.size() > 200);
    supergvins::capMatchesInPlace(geometry_inliers, supergvins::kMaxActiveTrackCount);
    assert(geometry_inliers.size() == 200);
    assert(geometry_inliers.front().previous == 256 && geometry_inliers.back().previous == 7);

    std::vector<supergvins::Match> fewer;
    for (int i=0;i<37;++i) fewer.push_back({i,i,0.9f-i*0.001f});
    supergvins::capMatchesInPlace(fewer, supergvins::kMaxActiveTrackCount);
    assert(fewer.size() == 37);

    const auto tied = supergvins::selectMatches(
        std::vector<supergvins::Match>{{3,4,0.8f},{2,8,0.8f},{1,2,0.8f}}, 4, 9, 0.5f, 3);
    assert(tied.size() == 3);
    assert(tied[0].previous == 1 && tied[0].current == 2);
    assert(tied[1].previous == 2 && tied[1].current == 8);
    assert(tied[2].previous == 3 && tied[2].current == 4);

    int next_id = 300;
    const auto continuing = supergvins::advanceMatchedTrack(
        42, 5, 0.1f, -0.2f, 0.4f, 0.3f, 0.5, next_id);
    assert(continuing.id == 42 && continuing.age == 6 && next_id == 300);
    assert(std::abs(continuing.velocity_x - 0.6f) < 1e-6f);
    assert(std::abs(continuing.velocity_y - 1.0f) < 1e-6f);
    const auto promoted = supergvins::advanceMatchedTrack(
        -1, 1, -0.2f, 0.1f, 0.1f, 0.4f, 0.5, next_id);
    assert(promoted.id == 300 && promoted.age == 2 && next_id == 301);
    assert(std::abs(promoted.velocity_x - 0.6f) < 1e-6f);
    assert(std::abs(promoted.velocity_y - 0.6f) < 1e-6f);
    bool exhausted = false;
    int exhausted_next_id = (1 << 24);
    try {
        (void)supergvins::advanceMatchedTrack(-1, 1, 0.f, 0.f, 1.f, 1.f,
                                             1.0, exhausted_next_id);
    } catch (const std::runtime_error&) { exhausted = true; }
    assert(exhausted);

    bool rejected = false;
    try { supergvins::selectKeypoints({NAN},512); } catch (...) { rejected = true; }
    assert(rejected);
    std::cout << "frontend policy tests passed\n";
}
