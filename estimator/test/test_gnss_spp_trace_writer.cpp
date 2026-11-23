#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "utility/gnss_spp_trace_writer.h"

namespace
{

std::string readFile(const std::string &path)
{
    std::ifstream input(path.c_str());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

gnss_comm::SppTrace makeTrace()
{
    gnss_comm::SppTrace trace;
    trace.solver_status = "converged";
    trace.iteration_count = 1;
    trace.final_dx_norm = 0.25;

    gnss_comm::SppTraceRow row;
    row.iteration = 0;
    row.obs_gpst_sec = 1234.5;
    row.sat = 1;
    row.system = SYS_GPS;
    row.prn = 1;
    row.l1_index = 0;
    row.used_in_solve = true;
    row.exclude_reason = "used";
    row.psr_m = 20200000.0;
    row.psr_std_m = 0.4;
    row.cn0_dbhz = 45.0;
    row.freq_hz = FREQ1;
    row.pre_residual_m = 12.0;
    row.post_residual_m = 0.5;
    row.pre_az_rad = 1.0;
    row.pre_el_rad = 0.8;
    row.post_az_rad = 1.1;
    row.post_el_rad = 0.9;
    row.weight = 2.0;
    row.iono_delay_m = 3.0;
    row.trop_delay_m = 2.0;
    row.sv_pos << 1.0, 2.0, 3.0;
    row.sv_clock_bias_s = 1e-6;
    row.rcv_state_pre.setZero();
    row.rcv_state_post.setOnes();
    row.dx_norm = 0.25;
    trace.rows.push_back(row);
    return trace;
}

TEST(GnssSppTraceWriterTest, TruncatesOnceThenAppendsCsvRows)
{
    char directory_template[] = "/tmp/gvins_spp_trace_test_XXXXXX";
    const char *directory = mkdtemp(directory_template);
    ASSERT_NE(nullptr, directory);
    const std::string path = std::string(directory) + "/trace.csv";

    GnssSppTraceWriter writer;
    std::string error;
    ASSERT_TRUE(writer.configure(path, false, &error)) << error;
    ASSERT_TRUE(writer.append(makeTrace(), 7, &error)) << error;
    ASSERT_TRUE(writer.append(makeTrace(), 8, &error)) << error;

    const std::string csv = readFile(path);
    EXPECT_NE(std::string::npos, csv.find(
        "residual_phase,is_post_initialization,spp_call_id,iteration,obs_gpst_sec"));
    EXPECT_NE(std::string::npos, csv.find("initialization_spp,0,7,0,1234.5"));
    EXPECT_NE(std::string::npos, csv.find("initialization_spp,0,8,0,1234.5"));
    EXPECT_NE(std::string::npos, csv.find(
        "rcv_post_dt_gps_m,rcv_post_dt_glo_m,rcv_post_dt_gal_m,rcv_post_dt_bds_m"));
    EXPECT_EQ(3u, static_cast<unsigned>(std::count(csv.begin(), csv.end(), '\n')));

    std::remove(path.c_str());
    rmdir(directory);
}

TEST(GnssSppTraceWriterTest, RejectsUnwritableDestinationWithoutThrowing)
{
    GnssSppTraceWriter writer;
    std::string error;
    EXPECT_FALSE(writer.configure("/proc/gvins/trace.csv", false, &error));
    EXPECT_FALSE(error.empty());
}

TEST(GnssSppTraceWriterTest, CreatesMissingParentDirectories)
{
    char directory_template[] = "/tmp/gvins_spp_trace_parent_test_XXXXXX";
    const char *directory = mkdtemp(directory_template);
    ASSERT_NE(nullptr, directory);
    const std::string nested_directory = std::string(directory) + "/one/two";
    const std::string path = nested_directory + "/trace.csv";

    GnssSppTraceWriter writer;
    std::string error;
    ASSERT_TRUE(writer.configure(path, false, &error)) << error;

    struct stat file_info;
    EXPECT_EQ(0, stat(path.c_str(), &file_info));

    std::remove(path.c_str());
    rmdir(nested_directory.c_str());
    rmdir((std::string(directory) + "/one").c_str());
    rmdir(directory);
}

}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
