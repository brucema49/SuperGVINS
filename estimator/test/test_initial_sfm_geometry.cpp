#include "initial/initial_sfm.h"
#include <cassert>
#include <iostream>
#include <limits>

int main()
{
    ceres::Solver::Summary summary;
    // Ceres leaves final_cost=-1 when it fails before evaluating a solution.
    assert(!acceptSfmSolution(summary));
    summary.termination_type = ceres::FAILURE;
    summary.final_cost = 1e-3;
    assert(!acceptSfmSolution(summary));
    summary.termination_type = ceres::CONVERGENCE;
    summary.final_cost = std::numeric_limits<double>::quiet_NaN();
    assert(!acceptSfmSolution(summary));
    summary.final_cost = std::numeric_limits<double>::infinity();
    assert(!acceptSfmSolution(summary));
    summary.termination_type = ceres::NO_CONVERGENCE;
    summary.final_cost = 1e-3;
    assert(acceptSfmSolution(summary));
    summary.final_cost = 1.0;
    assert(!acceptSfmSolution(summary));
    summary.termination_type = ceres::CONVERGENCE;
    assert(acceptSfmSolution(summary));

    std::vector<SFMFeature> features;
    for (int i = 0; i < 30; ++i) {
        SFMFeature f{};
        f.id = i;
        const Eigen::Vector3d point((i % 6 - 2) * 0.4, (i / 6 - 2) * 0.3, 4.0 + (i % 4));
        for (int frame = 0; frame < 3; ++frame) {
            Eigen::Vector3d camera_point = point - Eigen::Vector3d(frame * 0.5, 0, 0);
            f.observation.emplace_back(frame, camera_point.head<2>() / camera_point.z());
        }
        features.push_back(f);
    }
    // Parallel rays represent an infinite point; a reversed disparity is behind the cameras.
    for (int id = 30; id < 33; ++id) {
        SFMFeature f{};
        f.id = id;
        for (int frame = 0; frame < 3; ++frame)
            f.observation.emplace_back(frame, Eigen::Vector2d(0.2 + (id == 31 ? frame * 0.1 : (id == 32 ? -frame * 1e-5 : 0)), 0.2));
        features.push_back(f);
    }
    Eigen::Quaterniond rotations[3];
    Eigen::Vector3d translations[3];
    std::map<int, Eigen::Vector3d> points;
    GlobalSFM sfm;
    assert(sfm.construct(3, rotations, translations, 0, Eigen::Matrix3d::Identity(),
                         Eigen::Vector3d(1,0,0), features, points));
    for (int i = 0; i < 30; ++i) {
        assert(features[i].state && points.count(i));
        assert(points.at(i).allFinite() && points.at(i).z() > 0);
    }
    assert(!features[30].state && !points.count(30));
    assert(!features[31].state && !points.count(31));
    assert(!features[32].state && !points.count(32));
    std::cout << "initial SFM geometry tests passed\n";
}
