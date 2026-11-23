#include <stdio.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_utility.hpp>
#include <gvins/LocalSensorExternalTrigger.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

using namespace gnss_comm;

#define MAX_GNSS_CAMERA_DELAY 0.05

std::unique_ptr<Estimator> estimator_ptr;

std::condition_variable con;
std::condition_variable inflight_done;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;

struct FeatureMeasurement
{
    sensor_msgs::PointCloudConstPtr message;
    uint64_t generation = 0;
    uint64_t sequence = 0;
    int phase = -1;
    int64_t stamp_ns = -1;
    double stamp = std::numeric_limits<double>::quiet_NaN();
    size_t point_count = 0;
    std::set<std::string> wait_sync_reasons;
};

queue<FeatureMeasurement> feature_buf;
queue<std::vector<ObsPtr>> gnss_meas_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = -1;

std::mutex m_time;
std::mutex m_backend_csv;
std::ofstream backend_csv;
ros::Publisher reset_ack_pub;
std::atomic<bool> process_stop_requested(false);
std::atomic<uint64_t> reset_generation(0);
uint64_t feature_sequence = 0;
bool inflight_feature_valid = false;  // Protected by m_buf.
FeatureMeasurement inflight_feature;
int64_t last_valid_feature_stamp_ns = -1;  // ROS callbacks are serialized by ros::spin().
double next_pulse_time = 0.0;
bool next_pulse_time_valid;
double time_diff_gnss_local = 0.0;
bool time_diff_valid;
double latest_gnss_time;
double tmp_last_feature_time;
uint64_t feature_msg_counter;
int skip_parameter;

std::string csvEscape(const std::string &value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos)
        return value;
    std::string escaped = "\"";
    for (char c : value)
    {
        if (c == '"')
            escaped += '"';
        escaped += c;
    }
    escaped += '"';
    return escaped;
}

int64_t steadyClockNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void initializeBackendCsv()
{
    const size_t slash = VINS_RESULT_PATH.find_last_of('/');
    const std::string output_dir = slash == std::string::npos ? "." : VINS_RESULT_PATH.substr(0, slash);
    const std::string path = output_dir + "/backend.csv";
    std::lock_guard<std::mutex> lock(m_backend_csv);
    backend_csv.open(path.c_str(), std::ios::out | std::ios::trunc);
    if (!backend_csv.is_open())
    {
        ROS_WARN("Unable to open backend CSV at %s", path.c_str());
        return;
    }
    backend_csv << "schema_version,event_time_steady_ns,event,generation,sequence,feature_stamp_ns,feature_stamp,phase,point_count,gnss_enabled,solver_nonlinear,queue_depth,detail\n";
    backend_csv.flush();
    ROS_INFO("backend trace path %s", path.c_str());
}

void logBackendEvent(const std::string &event,
                     const FeatureMeasurement *feature = nullptr,
                     int solver_nonlinear = -1,
                     int queue_depth = -1,
                     const std::string &detail = "",
                     uint64_t generation = std::numeric_limits<uint64_t>::max())
{
    std::lock_guard<std::mutex> lock(m_backend_csv);
    if (!backend_csv.is_open())
        return;

    if (generation == std::numeric_limits<uint64_t>::max())
        generation = feature ? feature->generation : reset_generation.load();
    const uint64_t sequence = feature ? feature->sequence : 0;
    const int64_t stamp_ns = feature ? feature->stamp_ns : -1;
    const double stamp = feature ? feature->stamp : std::numeric_limits<double>::quiet_NaN();
    const int phase = feature ? feature->phase : -1;
    const size_t point_count = feature ? feature->point_count : 0;

    backend_csv << 1 << ',' << steadyClockNs() << ',' << event << ',' << generation << ','
                << sequence << ',' << stamp_ns << ',' << std::setprecision(15) << stamp << ','
                << phase << ',' << point_count << ',' << (GNSS_ENABLE ? 1 : 0) << ','
                << solver_nonlinear << ',' << queue_depth << ',' << csvEscape(detail) << '\n';
    backend_csv.flush();
}

FeatureMeasurement makeFeatureMeasurement(const sensor_msgs::PointCloudConstPtr &message,
                                          uint64_t sequence,
                                          int phase,
                                          uint64_t generation)
{
    FeatureMeasurement feature;
    feature.message = message;
    feature.sequence = sequence;
    feature.phase = phase;
    feature.generation = generation;
    feature.stamp = message->header.stamp.toSec();
    feature.stamp_ns = static_cast<int64_t>(message->header.stamp.toNSec());
    feature.point_count = message->points.size();
    return feature;
}

bool validateFeatureMessage(const sensor_msgs::PointCloud &message, std::string &reason)
{
    if (!std::isfinite(message.header.stamp.toSec()))
    {
        reason = "invalid_stamp";
        return false;
    }

    // Empty observations are valid phase-bearing messages. They do not need
    // channels because they will never reach processImage().
    if (message.points.empty())
    {
        if (message.channels.empty())
            return true;
        if (message.channels.size() < 5)
        {
            reason = "channel_count";
            return false;
        }
        for (size_t channel = 0; channel < 5; ++channel)
        {
            if (!message.channels[channel].values.empty())
            {
                reason = "channel_length_" + std::to_string(channel);
                return false;
            }
        }
        return true;
    }

    const size_t required_channels = 5;
    if (message.channels.size() < required_channels)
    {
        reason = "channel_count";
        return false;
    }
    for (size_t channel = 0; channel < required_channels; ++channel)
    {
        if (message.channels[channel].values.size() != message.points.size())
        {
            reason = "channel_length_" + std::to_string(channel);
            return false;
        }
    }

    std::set<int> seen_ids;
    for (size_t i = 0; i < message.points.size(); ++i)
    {
        const geometry_msgs::Point32 &point = message.points[i];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
            !std::isfinite(message.channels[1].values[i]) ||
            !std::isfinite(message.channels[2].values[i]) ||
            !std::isfinite(message.channels[3].values[i]) ||
            !std::isfinite(message.channels[4].values[i]))
        {
            reason = "nonfinite_value";
            return false;
        }
        if (point.z != 1.0f)
        {
            reason = "z_not_one";
            return false;
        }

        const double encoded_id = message.channels[0].values[i];
        if (!std::isfinite(encoded_id))
        {
            reason = "nonfinite_id";
            return false;
        }
        const double rounded_id = std::floor(encoded_id + 0.5);
        if (encoded_id < 0.0 || rounded_id < 0.0 ||
            rounded_id > static_cast<double>(std::numeric_limits<int>::max()) ||
            encoded_id > 16777216.0 || encoded_id != rounded_id)
        {
            reason = "invalid_id";
            return false;
        }
        const int value = static_cast<int>(rounded_id);
        const int feature_id = value / NUM_OF_CAM;
        const int camera_id = value % NUM_OF_CAM;
        if (feature_id < 0 || camera_id < 0 || camera_id >= NUM_OF_CAM)
        {
            reason = "invalid_id";
            return false;
        }
        if (!seen_ids.insert(value).second)
        {
            reason = "duplicate_id";
            return false;
        }
    }
    return true;
}

void logSyncWait(FeatureMeasurement &feature, const std::string &reason, int queue_depth)
{
    if (feature.wait_sync_reasons.insert(reason).second)
        logBackendEvent("wait_sync", &feature, -1, queue_depth, reason);
}

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator_ptr->g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator_ptr->g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator_ptr->Ps[WINDOW_SIZE];
    tmp_Q = estimator_ptr->Rs[WINDOW_SIZE];
    tmp_V = estimator_ptr->Vs[WINDOW_SIZE];
    tmp_Ba = estimator_ptr->Bas[WINDOW_SIZE];
    tmp_Bg = estimator_ptr->Bgs[WINDOW_SIZE];
    acc_0 = estimator_ptr->acc_0;
    gyr_0 = estimator_ptr->gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

bool
getMeasurements(std::vector<sensor_msgs::ImuConstPtr> &imu_msg,
                sensor_msgs::PointCloudConstPtr &img_msg,
                std::vector<ObsPtr> &gnss_msg,
                FeatureMeasurement &measurement)
{
    if (feature_buf.empty())
        return false;

    FeatureMeasurement &front_measurement = feature_buf.front();
    const double front_feature_ts = front_measurement.message->header.stamp.toSec();
    if (imu_buf.empty())
    {
        logSyncWait(front_measurement, "imu_empty", static_cast<int>(feature_buf.size()));
        return false;
    }

    if (GNSS_ENABLE && gnss_meas_buf.empty())
    {
        logSyncWait(front_measurement, "gnss_empty", static_cast<int>(feature_buf.size()));
        return false;
    }

    // Drop images which precede the oldest available IMU, but re-check for an
    // empty queue before accessing front() after each pop.
    while (!feature_buf.empty())
    {
        FeatureMeasurement &candidate = feature_buf.front();
        const double candidate_ts = candidate.message->header.stamp.toSec();
        const double candidate_imu_target = candidate_ts + estimator_ptr->td;
        if (!(imu_buf.back()->header.stamp.toSec() > candidate_imu_target))
        {
            ++sum_of_wait;
            logSyncWait(candidate, "imu_behind", static_cast<int>(feature_buf.size()));
            return false;
        }

        const double front_imu_ts = imu_buf.front()->header.stamp.toSec();
        if (front_imu_ts > candidate_ts)
        {
            ROS_WARN("throw image before oldest imu: image %.9f, imu %.9f", candidate_ts, front_imu_ts);
            logBackendEvent("drop_sync_stale", &candidate, -1,
                            static_cast<int>(feature_buf.size()), "imu_newer_than_image");
            feature_buf.pop();
            if (feature_buf.empty())
                return false;
            if (GNSS_ENABLE && gnss_meas_buf.empty())
                return false;
            continue;
        }
        break;
    }
    if (feature_buf.empty())
        return false;

    FeatureMeasurement &candidate = feature_buf.front();
    const double candidate_ts = candidate.message->header.stamp.toSec();
    double sync_offset = 0.0;
    if (GNSS_ENABLE)
    {
        std::lock_guard<std::mutex> time_lock(m_time);
        sync_offset = time_diff_gnss_local;
    }
    const double feature_sync_ts = candidate_ts + sync_offset;
    if (GNSS_ENABLE)
    {
        // gnss_meas_callback filters empty vectors before enqueueing them.
        while (!gnss_meas_buf.empty())
        {
            const double front_gnss_ts = time2sec(gnss_meas_buf.front()[0]->time);
            if (front_gnss_ts >= feature_sync_ts - MAX_GNSS_CAMERA_DELAY)
                break;
            ROS_WARN("throw stale gnss measurement %.9f before image %.9f", front_gnss_ts, feature_sync_ts);
            logBackendEvent("drop_gnss_stale", &candidate, -1,
                            static_cast<int>(feature_buf.size()),
                            "gnss_stamp=" + std::to_string(front_gnss_ts));
            gnss_meas_buf.pop();
        }
        if (gnss_meas_buf.empty())
        {
            logSyncWait(candidate, "gnss_behind", static_cast<int>(feature_buf.size()));
            return false;
        }

        const double front_gnss_ts = time2sec(gnss_meas_buf.front()[0]->time);
        const double gnss_delta = front_gnss_ts - feature_sync_ts;
        if (std::abs(gnss_delta) < MAX_GNSS_CAMERA_DELAY)
        {
            gnss_msg = gnss_meas_buf.front();
            gnss_meas_buf.pop();
            logBackendEvent("gnss_paired", &candidate, -1,
                            static_cast<int>(feature_buf.size()),
                            "gnss_stamp=" + std::to_string(front_gnss_ts) +
                                ";delta=" + std::to_string(gnss_delta) +
                                ";observations=" + std::to_string(gnss_msg.size()));
        }
        else
        {
            logBackendEvent("gnss_unpaired", &candidate, -1,
                            static_cast<int>(feature_buf.size()),
                            "gnss_stamp=" + std::to_string(front_gnss_ts) +
                                ";delta=" + std::to_string(gnss_delta));
        }
    }

    measurement = candidate;
    img_msg = candidate.message;
    feature_buf.pop();
    logBackendEvent("consume", &measurement, -1, static_cast<int>(feature_buf.size()));

    const double imu_target = img_msg->header.stamp.toSec() + estimator_ptr->td;
    while (!imu_buf.empty() && imu_buf.front()->header.stamp.toSec() < imu_target)
    {
        imu_msg.emplace_back(imu_buf.front());
        imu_buf.pop();
    }
    if (imu_buf.empty())
    {
        // The back timestamp was checked before popping the feature; this is a
        // defensive guard in case the IMU queue changes in a future caller.
        logBackendEvent("wait_sync", &measurement, -1, static_cast<int>(feature_buf.size()), "imu_exhausted");
        return false;
    }
    imu_msg.emplace_back(imu_buf.front());
    return true;
}

void resetBackendState(const std::string &reason, const std::string &token,
                       bool strict_timestamp_rewind = false,
                       int64_t reset_stamp_ns = -1)
{
    size_t pending_features = 0;
    size_t pending_imus = 0;
    size_t pending_gnss = 0;
    bool inflight_at_reset = false;
    uint64_t inflight_sequence = 0;
    bool retained_time_offset_valid = false;
    const bool clock_sync_invalidated = strict_timestamp_rewind && GNSS_ENABLE && GNSS_LOCAL_ONLINE_SYNC;
    const uint64_t new_generation = reset_generation.fetch_add(1) + 1;

    {
        std::lock_guard<std::mutex> lock(m_buf);
        while (!feature_buf.empty())
        {
            logBackendEvent("reset_pending", &feature_buf.front(), -1,
                            static_cast<int>(feature_buf.size()), reason);
            feature_buf.pop();
            ++pending_features;
        }
        if (inflight_feature_valid)
        {
            inflight_at_reset = true;
            inflight_sequence = inflight_feature.sequence;
        }
        pending_imus = imu_buf.size();
        while (!imu_buf.empty())
            imu_buf.pop();
        pending_gnss = gnss_meas_buf.size();
        while (!gnss_meas_buf.empty())
            gnss_meas_buf.pop();
    }

    double retained_time_offset = 0.0;
    {
        std::lock_guard<std::mutex> lock(m_time);
        feature_msg_counter = 0;
        latest_gnss_time = -1;
        tmp_last_feature_time = -1;
        skip_parameter = GNSS_ENABLE ? -1 : 0;
        if (clock_sync_invalidated)
        {
            // A camera rewind can mark a new sensor clock epoch. Wait for a
            // fresh PPS/trigger pair before selecting or pairing GNSS data.
            time_diff_valid = false;
            next_pulse_time_valid = false;
            time_diff_gnss_local = 0.0;
        }
        retained_time_offset = time_diff_gnss_local;
        retained_time_offset_valid = time_diff_valid;
    }

    {
        std::lock_guard<std::mutex> estimator_lock(m_estimator);
        std::lock_guard<std::mutex> state_lock(m_state);
        estimator_ptr->clearState();
        estimator_ptr->setParameter();
        if (GNSS_ENABLE && retained_time_offset_valid)
            estimator_ptr->inputGNSSTimeDiff(retained_time_offset);
        current_time = -1;
        latest_time = -1;
        last_imu_t = 0;
        init_feature = false;
        init_imu = true;
        tmp_P.setZero();
        tmp_Q.setIdentity();
        tmp_V.setZero();
        tmp_Ba.setZero();
        tmp_Bg.setZero();
        acc_0.setZero();
        gyr_0.setZero();
    }

    // A dequeued pre-reset batch either completes before the reset or observes
    // the new generation and is logged as reset_drop. Do not acknowledge the
    // ordered barrier until that batch has reached one of those terminal paths.
    {
        std::unique_lock<std::mutex> lock(m_buf);
        inflight_done.wait(lock, [] { return !inflight_feature_valid; });
    }

    if (reason == "ordered_barrier" && reset_stamp_ns >= 0)
        last_valid_feature_stamp_ns = reset_stamp_ns;
    else if (reason == "legacy_restart_bool")
        last_valid_feature_stamp_ns = -1;

    std::ostringstream detail;
    detail << "reason=" << reason << ";pending_features=" << pending_features
           << ";pending_imus=" << pending_imus << ";pending_gnss=" << pending_gnss
           << ";inflight_at_reset=" << (inflight_at_reset ? 1 : 0)
           << ";inflight_sequence=" << inflight_sequence
           << ";phase=" << (GNSS_ENABLE ? "recalibrate" : "even")
           << ";time_offset_retained=" << std::setprecision(15) << retained_time_offset
           << ";time_offset_valid=" << (retained_time_offset_valid ? 1 : 0)
           << ";strict_timestamp_rewind=" << (strict_timestamp_rewind ? 1 : 0)
           << ";clock_sync_invalidated=" << (clock_sync_invalidated ? 1 : 0)
           << ";clock_sync_invalidation_reason="
           << (clock_sync_invalidated ? "strict_timestamp_rewind" : "none");
    logBackendEvent("reset", nullptr, -1, 0, detail.str(), new_generation);

    if (!token.empty())
    {
        std_msgs::String ack;
        ack.data = token;
        reset_ack_pub.publish(ack);
        logBackendEvent("reset_ack", nullptr, -1, 0, "token=" + token, new_generation);
    }
    con.notify_all();
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

void gnss_ephem_callback(const GnssEphemMsgConstPtr &ephem_msg)
{
    EphemPtr ephem = msg2ephem(ephem_msg);
    std::lock_guard<std::mutex> lock(m_estimator);
    estimator_ptr->inputEphem(ephem);
}

void gnss_glo_ephem_callback(const GnssGloEphemMsgConstPtr &glo_ephem_msg)
{
    GloEphemPtr glo_ephem = msg2glo_ephem(glo_ephem_msg);
    std::lock_guard<std::mutex> lock(m_estimator);
    estimator_ptr->inputEphem(glo_ephem);
}

void gnss_iono_params_callback(const StampedFloat64ArrayConstPtr &iono_msg)
{
    double ts = iono_msg->header.stamp.toSec();
    std::vector<double> iono_params;
    std::copy(iono_msg->data.begin(), iono_msg->data.end(), std::back_inserter(iono_params));
    assert(iono_params.size() == 8);
    std::lock_guard<std::mutex> lock(m_estimator);
    estimator_ptr->inputIonoParams(ts, iono_params);
}

void gnss_meas_callback(const GnssMeasMsgConstPtr &meas_msg)
{
    std::vector<ObsPtr> gnss_meas = msg2meas(meas_msg);

    if (gnss_meas.empty())
    {
        logBackendEvent("gnss_rejected", nullptr, -1, -1, "empty_observations");
        return;
    }

    const double gnss_stamp = time2sec(gnss_meas[0]->time);
    bool sync_valid = false;
    {
        std::lock_guard<std::mutex> lock(m_time);
        latest_gnss_time = gnss_stamp;
        sync_valid = time_diff_valid;
    }

    FeatureMeasurement gnss_trace;
    gnss_trace.generation = reset_generation.load();
    gnss_trace.stamp = gnss_stamp;
    gnss_trace.stamp_ns = static_cast<int64_t>(std::llround(gnss_stamp * 1e9));
    gnss_trace.point_count = gnss_meas.size();
    logBackendEvent("gnss_received", &gnss_trace, -1, -1,
                    sync_valid ? "sync_valid=1" : "sync_valid=0");

    // cerr << "gnss ts is " << std::setprecision(20) << time2sec(gnss_meas[0]->time) << endl;
    if (!sync_valid)
        return;

    m_buf.lock();
    gnss_meas_buf.push(std::move(gnss_meas));
    m_buf.unlock();
    con.notify_one();
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    const std::string barrier_prefix = "supergvins_reset_barrier:";
    if (feature_msg->header.frame_id.compare(0, barrier_prefix.size(), barrier_prefix) == 0)
    {
        const std::string token = feature_msg->header.frame_id.substr(barrier_prefix.size());
        FeatureMeasurement barrier = makeFeatureMeasurement(feature_msg, 0, -1, reset_generation.load());
        if (token.empty() || !feature_msg->points.empty())
        {
            logBackendEvent("reset_barrier_rejected", &barrier, -1, -1,
                            token.empty() ? "empty_token" : "barrier_has_points");
            return;
        }
        logBackendEvent("reset_barrier_received", &barrier, -1, -1, "token=" + token);
        const bool strict_timestamp_rewind =
            last_valid_feature_stamp_ns >= 0 && barrier.stamp_ns < last_valid_feature_stamp_ns;
        resetBackendState("ordered_barrier", token, strict_timestamp_rewind, barrier.stamp_ns);
        return;
    }

    ++feature_msg_counter;
    const uint64_t sequence = ++feature_sequence;
    const int phase = static_cast<int>(feature_msg_counter % 2);
    FeatureMeasurement feature = makeFeatureMeasurement(feature_msg, sequence, phase, reset_generation.load());
    logBackendEvent("received", &feature, -1, -1);

    std::string rejection_reason;
    if (!validateFeatureMessage(*feature_msg, rejection_reason))
    {
        logBackendEvent("rejected", &feature, -1, -1, rejection_reason);
        return;
    }
    // Only valid observations advance the per-generation timestamp watermark;
    // empty observations are valid and still carry camera clock evidence.
    if (last_valid_feature_stamp_ns < 0 || feature.stamp_ns > last_valid_feature_stamp_ns)
        last_valid_feature_stamp_ns = feature.stamp_ns;

    {
        std::lock_guard<std::mutex> lock(m_time);
        if (skip_parameter < 0 && time_diff_valid)
        {
            const double this_feature_ts = feature_msg->header.stamp.toSec() + time_diff_gnss_local;
            if (latest_gnss_time > 0 && tmp_last_feature_time > 0)
            {
                if (std::abs(this_feature_ts - latest_gnss_time) > std::abs(tmp_last_feature_time - latest_gnss_time))
                    skip_parameter = static_cast<int>(feature_msg_counter % 2);  // skip this frame and afterwards
                else
                    skip_parameter = 1 - static_cast<int>(feature_msg_counter % 2);  // skip next frame and afterwards
            }
            // cerr << "feature counter is " << feature_msg_counter << ", skip parameter is " << int(skip_parameter) << endl;
            tmp_last_feature_time = this_feature_ts;
        }
    }

    if (feature_msg->points.empty())
        logBackendEvent("empty", &feature, -1, -1);

    int selected_skip_parameter;
    bool sync_valid;
    {
        std::lock_guard<std::mutex> lock(m_time);
        selected_skip_parameter = skip_parameter;
        sync_valid = time_diff_valid;
    }
    if (selected_skip_parameter < 0)
    {
        logBackendEvent("wait_sync", &feature, -1, -1,
                        sync_valid ? "phase_uncalibrated" : "clock_uncalibrated");
        return;
    }
    if (phase == selected_skip_parameter)
    {
        logBackendEvent("skip_parity", &feature, -1, -1,
                        "skip_parameter=" + std::to_string(selected_skip_parameter));
        return;
    }

    // Empty observations count toward phase selection but never initialize or
    // update the estimator.
    if (feature_msg->points.empty())
        return;

    {
        std::lock_guard<std::mutex> lock(m_buf);
        feature.generation = reset_generation.load();
        feature_buf.push(feature);
        logBackendEvent("enqueue", &feature, -1, static_cast<int>(feature_buf.size()));
    }
    con.notify_one();
}

void local_trigger_info_callback(const gvins::LocalSensorExternalTriggerConstPtr &trigger_msg)
{
    std::lock_guard<std::mutex> time_lock(m_time);

    if (next_pulse_time_valid)
    {
        time_diff_gnss_local = next_pulse_time - trigger_msg->header.stamp.toSec();
        {
            std::lock_guard<std::mutex> estimator_lock(m_estimator);
            estimator_ptr->inputGNSSTimeDiff(time_diff_gnss_local);
        }
        if (!time_diff_valid)       // just get calibrated
            std::cout << "time difference between GNSS and VI-Sensor got calibrated: "
                << std::setprecision(15) << time_diff_gnss_local << " s\n";
        time_diff_valid = true;
        logBackendEvent("sync_calibrated", nullptr, -1, -1,
                        "time_offset=" + std::to_string(time_diff_gnss_local));
    }
}

void gnss_tp_info_callback(const GnssTimePulseInfoMsgConstPtr &tp_msg)
{
    gtime_t tp_time = gpst2time(tp_msg->time.week, tp_msg->time.tow);
    if (tp_msg->utc_based || tp_msg->time_sys == SYS_GLO)
        tp_time = utc2gpst(tp_time);
    else if (tp_msg->time_sys == SYS_GAL)
        tp_time = gst2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_BDS)
        tp_time = bdt2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_NONE)
    {
        std::cerr << "Unknown time system in GNSSTimePulseInfoMsg.\n";
        return;
    }
    double gnss_ts = time2sec(tp_time);

    std::lock_guard<std::mutex> lg(m_time);
    next_pulse_time = gnss_ts;
    next_pulse_time_valid = true;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        resetBackendState("legacy_restart_bool", "");
    }
    return;
}

void process()
{
    while (true)
    {
        std::vector<sensor_msgs::ImuConstPtr> imu_msg;
        sensor_msgs::PointCloudConstPtr img_msg;
        std::vector<ObsPtr> gnss_msg;
        FeatureMeasurement measurement;
        bool got_measurement = false;

        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
                    if (process_stop_requested.load())
                        return true;
                    got_measurement = getMeasurements(imu_msg, img_msg, gnss_msg, measurement);
                    if (got_measurement)
                    {
                        inflight_feature = measurement;
                        inflight_feature_valid = true;
                    }
                    return got_measurement;
                 });
        if (process_stop_requested.load())
        {
            if (got_measurement)
            {
                logBackendEvent("shutdown_pending", &measurement, -1, static_cast<int>(feature_buf.size()),
                                "consumed_but_not_started");
                inflight_feature_valid = false;
                inflight_done.notify_all();
            }
            lk.unlock();
            break;
        }
        lk.unlock();

        m_estimator.lock();
        if (process_stop_requested.load())
        {
            m_estimator.unlock();
            logBackendEvent("shutdown_pending", &measurement, -1, -1, "consumed_but_not_started");
            std::lock_guard<std::mutex> lock(m_buf);
            inflight_feature_valid = false;
            inflight_done.notify_all();
            break;
        }
        if (measurement.generation != reset_generation.load())
        {
            m_estimator.unlock();
            logBackendEvent("reset_drop", &measurement, -1, -1, "stale_generation");
            std::lock_guard<std::mutex> lock(m_buf);
            inflight_feature_valid = false;
            inflight_done.notify_all();
            continue;
        }

        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
        for (auto &imu_data : imu_msg)
        {
            double t = imu_data->header.stamp.toSec();
            double img_t = img_msg->header.stamp.toSec() + estimator_ptr->td;
            if (t <= img_t)
            { 
                if (current_time < 0)
                    current_time = t;
                double dt = t - current_time;
                ROS_ASSERT(dt >= 0);
                current_time = t;
                dx = imu_data->linear_acceleration.x;
                dy = imu_data->linear_acceleration.y;
                dz = imu_data->linear_acceleration.z;
                rx = imu_data->angular_velocity.x;
                ry = imu_data->angular_velocity.y;
                rz = imu_data->angular_velocity.z;
                estimator_ptr->processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

            }
            else
            {
                double dt_1 = img_t - current_time;
                double dt_2 = t - img_t;
                current_time = img_t;
                ROS_ASSERT(dt_1 >= 0);
                ROS_ASSERT(dt_2 >= 0);
                ROS_ASSERT(dt_1 + dt_2 > 0);
                double w1 = dt_2 / (dt_1 + dt_2);
                double w2 = dt_1 / (dt_1 + dt_2);
                dx = w1 * dx + w2 * imu_data->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_data->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_data->linear_acceleration.z;
                rx = w1 * rx + w2 * imu_data->angular_velocity.x;
                ry = w1 * ry + w2 * imu_data->angular_velocity.y;
                rz = w1 * rz + w2 * imu_data->angular_velocity.z;
                estimator_ptr->processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
            }
        }

        if (GNSS_ENABLE && !gnss_msg.empty())
            estimator_ptr->processGNSS(gnss_msg);

        ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

        TicToc t_s;
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
        for (unsigned int i = 0; i < img_msg->points.size(); i++)
        {
            int v = img_msg->channels[0].values[i] + 0.5;
            int feature_id = v / NUM_OF_CAM;
            int camera_id = v % NUM_OF_CAM;
            double x = img_msg->points[i].x;
            double y = img_msg->points[i].y;
            double z = img_msg->points[i].z;
            double p_u = img_msg->channels[1].values[i];
            double p_v = img_msg->channels[2].values[i];
            double velocity_x = img_msg->channels[3].values[i];
            double velocity_y = img_msg->channels[4].values[i];
            ROS_ASSERT(z == 1);
            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
        estimator_ptr->processImage(image, img_msg->header);

        const bool optimized = estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR;
        logBackendEvent("processed", &measurement, optimized ? 1 : 0, -1);
        if (optimized)
            logBackendEvent("optimized", &measurement, 1, -1, "solver_flag=NON_LINEAR");

        double whole_t = t_s.toc();
        printStatistics(*estimator_ptr, whole_t);
        std_msgs::Header header = img_msg->header;
        header.frame_id = "world";

        pubOdometry(*estimator_ptr, header);
        pubKeyPoses(*estimator_ptr, header);
        pubCameraPose(*estimator_ptr, header);
        pubPointCloud(*estimator_ptr, header);
        pubTF(*estimator_ptr, header);
        pubKeyframe(*estimator_ptr);
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        inflight_feature_valid = false;
        m_buf.unlock();
        inflight_done.notify_all();
    }
}

size_t logShutdownPending()
{
    size_t pending_feature_count = 0;
    size_t pending_imu_count = 0;
    size_t pending_gnss_count = 0;
    {
        std::lock_guard<std::mutex> lock(m_buf);
        queue<FeatureMeasurement> pending_features = feature_buf;
        pending_feature_count = pending_features.size();
        while (!pending_features.empty())
        {
            logBackendEvent("shutdown_pending", &pending_features.front(), -1,
                            static_cast<int>(feature_buf.size()), "queued_at_shutdown");
            pending_features.pop();
        }
        if (inflight_feature_valid)
        {
            logBackendEvent("shutdown_pending", &inflight_feature, -1, 0, "inflight_at_shutdown");
            ++pending_feature_count;
        }
        pending_imu_count = imu_buf.size();
        pending_gnss_count = gnss_meas_buf.size();
    }

    std::ostringstream detail;
    detail << "pending_features=" << pending_feature_count
           << ";pending_imus=" << pending_imu_count
           << ";pending_gnss=" << pending_gnss_count;
    logBackendEvent("shutdown", nullptr, -1, static_cast<int>(pending_feature_count), detail.str());
    return pending_feature_count;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "supergvins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    initializeBackendCsv();
    estimator_ptr.reset(new Estimator());
    estimator_ptr->setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    registerPub(n);
    reset_ack_pub = n.advertise<std_msgs::String>("/supergvins/reset_ack", 10, false);

    next_pulse_time_valid = false;
    time_diff_valid = false;
    latest_gnss_time = -1;
    tmp_last_feature_time = -1;
    feature_msg_counter = 0;

    if (GNSS_ENABLE)
        skip_parameter = -1;
    else
        skip_parameter = 0;

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_feature = n.subscribe("/supergvins_feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/supergvins_feature_tracker/restart", 2000, restart_callback);

    ros::Subscriber sub_ephem, sub_glo_ephem, sub_gnss_meas, sub_gnss_iono_params;
    ros::Subscriber sub_gnss_time_pluse_info, sub_local_trigger_info;
    if (GNSS_ENABLE)
    {
        sub_ephem = n.subscribe(GNSS_EPHEM_TOPIC, 100, gnss_ephem_callback);
        sub_glo_ephem = n.subscribe(GNSS_GLO_EPHEM_TOPIC, 100, gnss_glo_ephem_callback);
        sub_gnss_meas = n.subscribe(GNSS_MEAS_TOPIC, 100, gnss_meas_callback);
        sub_gnss_iono_params = n.subscribe(GNSS_IONO_PARAMS_TOPIC, 100, gnss_iono_params_callback);

        if (GNSS_LOCAL_ONLINE_SYNC)
        {
            sub_gnss_time_pluse_info = n.subscribe(GNSS_TP_INFO_TOPIC, 100, 
                gnss_tp_info_callback);
            sub_local_trigger_info = n.subscribe(LOCAL_TRIGGER_INFO_TOPIC, 100, 
                local_trigger_info_callback);
        }
        else
        {
            time_diff_gnss_local = GNSS_LOCAL_TIME_DIFF;
            estimator_ptr->inputGNSSTimeDiff(time_diff_gnss_local);
            time_diff_valid = true;
        }
    }

    std::thread measurement_process{process};
    ros::spin();

    {
        // Share the condition-variable mutex while setting the predicate so a
        // notification cannot be lost between the wait predicate and sleep.
        std::lock_guard<std::mutex> lock(m_buf);
        process_stop_requested.store(true);
    }
    con.notify_all();
    measurement_process.join();
    const size_t unprocessed_features = logShutdownPending();
    {
        std::lock_guard<std::mutex> lock(m_backend_csv);
        if (backend_csv.is_open())
        {
            backend_csv.flush();
            backend_csv.close();
        }
    }
    ROS_INFO("measurement process stopped gracefully; unprocessed feature queue entries: %zu",
             unprocessed_features);

    return 0;
}
