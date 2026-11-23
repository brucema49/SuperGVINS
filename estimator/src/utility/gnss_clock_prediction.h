#pragma once

// Receiver clock bias is measured in metres and drift in metres per second.
inline double predictGnssClockBias(double bias_m, double drift_mps,
    double reference_time_s, double target_time_s)
{
    return bias_m + drift_mps * (target_time_s - reference_time_s);
}
