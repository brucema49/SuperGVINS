#include "frontend_policy.h"
#include <cassert>
#include <iostream>

int main()
{
    using namespace supergvins;
    // Strong new pair matches must not evict geometrically verified long tracks.
    std::vector<Match> matches{{0,0,0.99f}, {1,1,0.95f}, {2,2,0.80f},
                               {3,3,0.70f}, {4,4,0.90f}};
    capMatchesByTrackAge(matches, 3, {0, 1, 12, 12, 2});
    assert(matches.size() == 3);
    assert(matches[0].previous == 2 && matches[1].previous == 3);
    assert(matches[2].previous == 4);
    // With no established tracks, retain score order and deterministic ties.
    matches = {{2,2,0.80f}, {0,0,0.90f}, {1,1,0.90f}};
    capMatchesByTrackAge(matches, 2, {0, 0, 0});
    assert(matches.size() == 2 && matches[0].previous == 0 && matches[1].previous == 1);
    // Do not invent matches when geometry has fewer inliers than the budget.
    matches = {{0,0,0.80f}};
    capMatchesByTrackAge(matches, 200, {8});
    assert(matches.size() == 1 && matches[0].previous == 0);
    capMatchesByTrackAge(matches, 0, {8});
    assert(matches.empty());
    std::cout << "track continuity tests passed\n";
}
