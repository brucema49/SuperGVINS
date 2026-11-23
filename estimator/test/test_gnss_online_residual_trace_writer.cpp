#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include <gnss_comm/gnss_constant.hpp>
#include <gnss_comm/gnss_utility.hpp>

#include "factor/gnss_psr_dopp_factor.hpp"
#include "utility/gnss_online_residual_trace_writer.h"

using ResidualSnapshotMethod = bool (GnssPsrDoppFactor::*)(
    double const *const *, GnssOnlineResidualSnapshot &) const;
static_assert(std::is_same<
    decltype(&GnssPsrDoppFactor::EvaluateResidualSnapshot),
    ResidualSnapshotMethod>::value,
    "GNSS factor residual snapshot API must remain read-only");

namespace
{

std::string readFile(const std::string &path)
{
    std::ifstream input(path.c_str());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

GnssOnlineResidualTraceRow makeRow()
{
    GnssOnlineResidualTraceRow row;
    row.window_index = 4;
    row.lower_index = 3;
    row.upper_index = 4;
    row.interpolation_ratio = 0.25;
    row.obs_gpst_sec = 1234.5;
    row.sat = 1;
    row.system = SYS_GPS;
    row.prn = 1;
    row.l1_index = 0;
    row.psr_m = 20200000.0;
    row.psr_std_m = 0.4;
    row.cn0_dbhz = 45.0;
    row.freq_hz = FREQ1;
    row.prior.raw_psr_residual_m = 12.0;
    row.prior.weighted_psr_residual = 3.0;
    row.prior.psr_weight = 0.25;
    row.prior.rcv_ecef << 1.0, 2.0, 3.0;
    row.prior.rcv_dt_m = 100.0;
    row.posterior.raw_psr_residual_m = 0.5;
    row.posterior.weighted_psr_residual = 0.1;
    row.posterior.psr_weight = 0.2;
    row.posterior.rcv_ecef << 4.0, 5.0, 6.0;
    row.posterior.rcv_dt_m = 101.0;
    return row;
}

GnssOnlineResidualSolverSummary makeSummary()
{
    GnssOnlineResidualSolverSummary summary;
    summary.termination_type = "CONVERGENCE";
    summary.iterations = 5;
    summary.successful_steps = 2;
    summary.initial_cost = 10.0;
    summary.final_cost = 2.0;
    return summary;
}

TEST(GnssOnlineResidualTraceWriterTest, TruncatesOnceThenAppendsRowsWithPhase)
{
    char directory_template[] = "/tmp/gvins_online_trace_test_XXXXXX";
    const char *directory = mkdtemp(directory_template);
    ASSERT_NE(nullptr, directory);
    const std::string path = std::string(directory) + "/trace.csv";

    std::string error;
    {
        GnssOnlineResidualTraceWriter writer;
        ASSERT_TRUE(writer.configure(path, false, &error)) << error;
        ASSERT_TRUE(writer.append(std::vector<GnssOnlineResidualTraceRow>(1, makeRow()),
                                  makeSummary(), 7, &error)) << error;
    }
    {
        GnssOnlineResidualTraceWriter append_writer;
        ASSERT_TRUE(append_writer.configure(path, true, &error)) << error;
        ASSERT_TRUE(append_writer.append(std::vector<GnssOnlineResidualTraceRow>(1, makeRow()),
                                         makeSummary(), 8, &error)) << error;
    }

    const std::string csv = readFile(path);
    EXPECT_NE(std::string::npos, csv.find(
        "residual_phase,is_post_initialization,optimization_call_id"));
    EXPECT_NE(std::string::npos, csv.find(
        "prior_raw_psr_residual_m,posterior_raw_psr_residual_m"));
    EXPECT_NE(std::string::npos, csv.find("post_initialization,1,7,4,3,4"));
    EXPECT_NE(std::string::npos, csv.find("post_initialization,1,8,4,3,4"));
    EXPECT_NE(std::string::npos, csv.find(
        "solver_termination_type,solver_iterations,solver_successful_steps,has_accepted_update,"));
    EXPECT_NE(std::string::npos, csv.find("CONVERGENCE,5,2,1,10,2"));
    EXPECT_EQ(3u, static_cast<unsigned>(std::count(csv.begin(), csv.end(), '\n')));
    std::istringstream lines(csv);
    for (std::string line; std::getline(lines, line); )
        EXPECT_EQ(36u, static_cast<unsigned>(std::count(line.begin(), line.end(), ',') + 1));

    std::remove(path.c_str());
    rmdir(directory);
}

TEST(GnssOnlineResidualTraceWriterTest, RejectsUnwritableDestination)
{
    GnssOnlineResidualTraceWriter writer;
    std::string error;
    EXPECT_FALSE(writer.configure("/proc/gvins/online_trace.csv", false, &error));
    EXPECT_FALSE(error.empty());
}

TEST(GnssOnlineResidualTraceWriterTest, CreatesMissingParentDirectories)
{
    char directory_template[] = "/tmp/gvins_online_trace_parent_test_XXXXXX";
    const char *directory = mkdtemp(directory_template);
    ASSERT_NE(nullptr, directory);
    const std::string nested_directory = std::string(directory) + "/one/two";
    const std::string path = nested_directory + "/trace.csv";

    GnssOnlineResidualTraceWriter writer;
    std::string error;
    ASSERT_TRUE(writer.configure(path, false, &error)) << error;

    struct stat file_info;
    EXPECT_EQ(0, stat(path.c_str(), &file_info));

    std::remove(path.c_str());
    rmdir(nested_directory.c_str());
    rmdir((std::string(directory) + "/one").c_str());
    rmdir(directory);
}

TEST(GnssOnlineResidualSnapshotTest, MatchesFactorEvaluateWithoutMutatingParameters)
{
    gnss_comm::ObsPtr obs(new gnss_comm::Obs());
    obs->time = gnss_comm::gpst2time(2200, 100000.0);
    obs->sat = gnss_comm::sat_no(SYS_GLO, 1);
    obs->freqs.push_back(FREQ1_GLO);
    obs->CN0.push_back(45.0);
    obs->psr.push_back(24000000.0);
    obs->psr_std.push_back(0.5);
    obs->dopp.push_back(0.0);
    obs->dopp_std.push_back(0.5);

    gnss_comm::GloEphemPtr ephem(new gnss_comm::GloEphem());
    ephem->sat = obs->sat;
    ephem->toe = obs->time;
    ephem->freqo = 0;
    ephem->pos[0] = 19100000.0;
    ephem->pos[1] = 13400000.0;
    ephem->pos[2] = 10100000.0;
    ephem->vel[0] = -1200.0;
    ephem->vel[1] = 2200.0;
    ephem->vel[2] = 900.0;
    ephem->acc[0] = ephem->acc[1] = ephem->acc[2] = 0.0;
    ephem->tau_n = 0.0;
    ephem->gamma = 0.0;

    std::vector<double> iono_parameters(8, 0.0);
    GnssPsrDoppFactor factor(obs, ephem, iono_parameters, 0.25);

    std::array<double, 7> pose_i = {{1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0}};
    std::array<double, 9> speed_i = {{0.0}};
    std::array<double, 7> pose_j = {{4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0}};
    std::array<double, 9> speed_j = {{0.0}};
    std::array<double, 1> rcv_dt = {{100.0}};
    std::array<double, 1> rcv_ddt = {{0.0}};
    std::array<double, 1> yaw = {{0.1}};
    std::array<double, 3> anchor = {{-2425943.0, 5380844.0, 2408961.0}};
    std::array<const double *, 8> parameters = {{pose_i.data(), speed_i.data(),
        pose_j.data(), speed_j.data(), rcv_dt.data(), rcv_ddt.data(), yaw.data(),
        anchor.data()}};

    const std::array<double, 7> pose_i_before = pose_i;
    const std::array<double, 1> rcv_dt_before = rcv_dt;
    double residuals[2] = {0.0, 0.0};
    ASSERT_TRUE(factor.Evaluate(parameters.data(), residuals, nullptr));
    GnssOnlineResidualSnapshot snapshot;
    ASSERT_TRUE(factor.EvaluateResidualSnapshot(parameters.data(), snapshot));

    EXPECT_DOUBLE_EQ(residuals[0], snapshot.weighted_psr_residual);
    EXPECT_NEAR(snapshot.raw_psr_residual_m * snapshot.psr_weight,
        snapshot.weighted_psr_residual, 1e-10);
    EXPECT_EQ(pose_i_before, pose_i);
    EXPECT_EQ(rcv_dt_before, rcv_dt);

    pose_i[0] += 10.0;
    rcv_dt[0] += 100.0;
    GnssOnlineResidualSnapshot changed_snapshot;
    ASSERT_TRUE(factor.EvaluateResidualSnapshot(parameters.data(), changed_snapshot));
    EXPECT_NE(snapshot.raw_psr_residual_m, changed_snapshot.raw_psr_residual_m);
    EXPECT_NE(snapshot.rcv_ecef.x(), changed_snapshot.rcv_ecef.x());
    EXPECT_DOUBLE_EQ(100.0, changed_snapshot.rcv_dt_m - snapshot.rcv_dt_m);
}

}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
