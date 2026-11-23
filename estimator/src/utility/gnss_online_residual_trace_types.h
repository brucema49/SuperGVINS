#pragma once

#include <cstdint>
#include <string>

#include <Eigen/Dense>

struct GnssOnlineResidualSnapshot
{
    double raw_psr_residual_m = 0.0;
    double weighted_psr_residual = 0.0;
    double psr_weight = 0.0;
    Eigen::Vector3d rcv_ecef = Eigen::Vector3d::Zero();
    double rcv_dt_m = 0.0;
};

struct GnssOnlineResidualTraceRow
{
    uint32_t window_index = 0;
    int lower_index = -1;
    int upper_index = -1;
    double interpolation_ratio = 0.0;
    double obs_gpst_sec = 0.0;
    uint32_t sat = 0;
    uint32_t system = 0;
    uint32_t prn = 0;
    int l1_index = -1;
    double psr_m = 0.0;
    double psr_std_m = 0.0;
    double cn0_dbhz = 0.0;
    double freq_hz = 0.0;
    GnssOnlineResidualSnapshot prior;
    GnssOnlineResidualSnapshot posterior;
};

struct GnssOnlineResidualSolverSummary
{
    std::string termination_type;
    int iterations = 0;
    int successful_steps = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
};
