#include "learned_frontend.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace {
std::unique_ptr<supergvins::LearnedFrontend> tracker;
ros::Publisher feature_pub;
ros::Subscriber* reset_ack_subscriber_ptr=nullptr;
double last_stamp = -1;
double last_rejected_duplicate_stamp = -1;
int width=0, height=0;
uint64_t frame_number=0;
uint64_t generation=0;
bool processing_failed=false;
std::ofstream frame_log;
std::mutex reset_ack_mutex;
std::condition_variable reset_ack_condition;
std::string expected_reset_ack;
bool reset_ack_received=false;

uint64_t steadyNowNs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void resetAckCallback(const std_msgs::StringConstPtr& message)
{
    std::lock_guard<std::mutex> lock(reset_ack_mutex);
    if (!expected_reset_ack.empty() && message->data == expected_reset_ack) {
        reset_ack_received=true;
        reset_ack_condition.notify_all();
    }
}

std::string resetToken()
{
    static std::atomic<uint64_t> token_counter{0};
    std::ostringstream token;
    token << ::getpid() << '-' << generation << '-' << steadyNowNs() << '-'
          << token_counter.fetch_add(1, std::memory_order_relaxed);
    return token.str();
}

bool resetDownstream(const ros::Time& stamp)
{
    tracker->reset();
    ++generation;
    last_rejected_duplicate_stamp=-1;

    if (reset_ack_subscriber_ptr == nullptr) {
        ROS_WARN_STREAM("local tracker reset generation=" << generation
                        << "; no downstream reset subscription is active");
        return true;
    }

    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    const auto protocol_probe_deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(250);
    while (ros::ok() && reset_ack_subscriber_ptr->getNumPublishers() == 0 &&
           std::chrono::steady_clock::now() < protocol_probe_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    // A frontend used without the estimator has no downstream state to reset.
    // Presence of the acknowledgement publisher identifies the coordinated
    // backend; feature topic subscribers may also include visualizers.
    if (ros::ok() && reset_ack_subscriber_ptr->getNumPublishers() == 0) {
        ROS_WARN_STREAM("local tracker reset generation=" << generation
                        << "; no /supergvins/reset_ack publisher is connected");
        return true;
    }
    while (ros::ok() && feature_pub.getNumSubscribers() == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!ros::ok() || feature_pub.getNumSubscribers() == 0) {
        ROS_FATAL_STREAM("reset barrier generation=" << generation
                         << " failed: feature subscriber did not connect within 5 s");
        return false;
    }

    const std::string token=resetToken();
    {
        std::lock_guard<std::mutex> lock(reset_ack_mutex);
        expected_reset_ack=token;
        reset_ack_received=false;
    }

    sensor_msgs::PointCloud barrier;
    barrier.header.stamp=stamp;
    barrier.header.frame_id="supergvins_reset_barrier:"+token;
    feature_pub.publish(barrier);

    bool acknowledged=false;
    {
        std::unique_lock<std::mutex> lock(reset_ack_mutex);
        acknowledged=reset_ack_condition.wait_until(lock,deadline,[] { return reset_ack_received; });
        expected_reset_ack.clear();
        reset_ack_received=false;
    }
    if (!acknowledged) {
        ROS_FATAL_STREAM("reset barrier generation=" << generation << " token=" << token
                         << " was not acknowledged within 5 s; stopping before publishing features");
        return false;
    }
    ROS_INFO_STREAM("reset barrier acknowledged generation=" << generation << " token=" << token);
    return true;
}

void imageCallback(const sensor_msgs::ImageConstPtr& message)
{
    const uint64_t receive_steady_ns=steadyNowNs();
    const double stamp=message->header.stamp.toSec();
    try {
        if (last_stamp >= 0 && stamp == last_stamp) {
            if (last_rejected_duplicate_stamp != stamp) {
                ROS_WARN_STREAM("rejecting duplicate image timestamp_ns=" << message->header.stamp.toNSec()
                                << " receive_steady_ns=" << receive_steady_ns
                                << "; resetting tracker and requesting estimator reset");
                if (!resetDownstream(message->header.stamp)) {
                    processing_failed=true;
                    ros::shutdown();
                }
                last_rejected_duplicate_stamp=stamp;
            } else {
                ROS_WARN_STREAM("rejecting repeated duplicate image timestamp_ns="
                                << message->header.stamp.toNSec() << " receive_steady_ns=" << receive_steady_ns
                                << " in generation=" << generation);
            }
            return;
        }
        if (last_stamp >= 0 && (stamp < last_stamp || stamp-last_stamp > 1.0)) {
            const bool rewound=stamp < last_stamp;
            ROS_WARN_STREAM("image timestamp " << (rewound ? "moved backwards" : "jumped by more than 1 s")
                            << " at timestamp_ns=" << message->header.stamp.toNSec()
                            << " receive_steady_ns=" << receive_steady_ns
                            << "; resetting tracker and requesting estimator reset");
            if (rewound) last_stamp=-1;
            if (!resetDownstream(message->header.stamp)) {
                processing_failed=true;
                ros::shutdown();
                return;
            }
        }
        if (last_rejected_duplicate_stamp >= 0 && stamp != last_rejected_duplicate_stamp)
            last_rejected_duplicate_stamp=-1;
        cv_bridge::CvImageConstPtr image;
        if (message->encoding == "8UC1") {
            sensor_msgs::Image copy=*message;
            copy.encoding=sensor_msgs::image_encodings::MONO8;
            image=cv_bridge::toCvCopy(copy,sensor_msgs::image_encodings::MONO8);
        } else image=cv_bridge::toCvShare(message,sensor_msgs::image_encodings::MONO8);
        if (image->image.cols!=width || image->image.rows!=height)
            throw std::runtime_error("image dimensions do not match calibration");
        supergvins::FrameStats stats;
        const auto tracks=tracker->process(image->image,stamp,stats);
        last_stamp=stamp;
        sensor_msgs::PointCloud cloud;
        cloud.header=message->header; cloud.header.frame_id="world";
        cloud.channels.resize(5);
        cloud.channels[0].name="id"; cloud.channels[1].name="u"; cloud.channels[2].name="v";
        cloud.channels[3].name="velocity_x"; cloud.channels[4].name="velocity_y";
        for (const auto& feature:tracks) {
            if (feature.age<=1) continue;
            geometry_msgs::Point32 point; point.x=feature.normalized.x; point.y=feature.normalized.y; point.z=1;
            cloud.points.push_back(point);
            cloud.channels[0].values.push_back(feature.id);
            cloud.channels[1].values.push_back(feature.pixel.x);
            cloud.channels[2].values.push_back(feature.pixel.y);
            cloud.channels[3].values.push_back(feature.velocity.x);
            cloud.channels[4].values.push_back(feature.velocity.y);
        }
        feature_pub.publish(cloud);
        const uint64_t publish_steady_ns=steadyNowNs();
        ++frame_number;
        if (frame_log) {
            frame_log << frame_number << ',' << message->header.stamp.toNSec() << ',' << stats.extractor_calls << ','
                      << stats.raw << ',' << stats.candidates << ',' << stats.previous << ',' << stats.previous_candidates << ','
                      << stats.raw_matches << ',' << stats.matches << ',' << stats.ransac_inliers_before_cap << ','
                      << stats.inliers << ',' << stats.retained << ',' << cloud.points.size() << ','
                      << stats.extract_ms << ',' << stats.match_ms << ',' << stats.total_ms << ','
                      << receive_steady_ns << ',' << publish_steady_ns << ',' << generation << '\n';
            frame_log.flush();
        }
        ROS_INFO_THROTTLE(5.0,"SuperPoint/LightGlue frame=%llu raw=%zu kept=%zu prev_candidates=%zu matches=%zu ransac=%zu capped=%zu tracks=%zu published=%zu",
                          static_cast<unsigned long long>(frame_number),stats.raw,stats.candidates,
                          stats.previous_candidates,stats.matches,stats.ransac_inliers_before_cap,
                          stats.inliers,stats.retained,cloud.points.size());
    } catch (const std::exception& error) {
        ROS_FATAL_STREAM("learned frontend stopped at timestamp_ns=" << message->header.stamp.toNSec()
                         << " receive_steady_ns=" << receive_steady_ns << " generation=" << generation
                         << ": " << error.what());
        processing_failed=true;
        ros::shutdown();
    }
}
}
int main(int argc,char** argv)
{
    ros::init(argc,argv,"supergvins_feature_tracker");
    ros::NodeHandle n("~");
    try {
        std::string config, folder;
        if (!n.getParam("config_file",config) || !n.getParam("gvins_folder",folder))
            throw std::runtime_error("config_file and gvins_folder parameters are required");
        cv::FileStorage fs(config,cv::FileStorage::READ);
        if (!fs.isOpened()) throw std::runtime_error("cannot open frontend configuration: "+config);
        std::string topic, extractor, matcher, device, output;
        fs["image_topic"] >> topic; fs["extractor_weight_path"] >> extractor;
        fs["matcher_weight_path"] >> matcher; fs["inference_device"] >> device; fs["output_dir"] >> output;
        width=static_cast<int>(fs["image_width"]); height=static_cast<int>(fs["image_height"]);
        double match_threshold=0.5, ransac_threshold=3.0;
        int min_distance=20;
        if (!fs["matcher_score_threshold"].empty()) match_threshold=static_cast<double>(fs["matcher_score_threshold"]);
        if (!fs["ransac_reproj_threshold"].empty()) ransac_threshold=static_cast<double>(fs["ransac_reproj_threshold"]);
        if (!fs["min_dist"].empty()) min_distance=static_cast<int>(fs["min_dist"]);
        if (topic.empty() || extractor.empty() || matcher.empty() || width<=0 || height<=0 || min_distance<0 || match_threshold<0 || match_threshold>=1 || ransac_threshold<=0)
            throw std::runtime_error("invalid frontend configuration");
        if (!fs["fisheye"].empty() && static_cast<int>(fs["fisheye"]) != 0)
            throw std::runtime_error("fisheye mask is not yet supported by learned frontend");
        // roslaunch can resolve $(find package)/.. via devel/share; use rospack's source package path.
        const std::string package_path=ros::package::getPath("supergvins_feature_tracker");
        if (package_path.empty()) throw std::runtime_error("cannot locate supergvins_feature_tracker package");
        const std::string model_root=package_path+"/..";
        if (extractor[0]!='/') extractor=model_root+"/"+extractor;
        if (matcher[0]!='/') matcher=model_root+"/"+matcher;
        tracker.reset(new supergvins::LearnedFrontend(extractor,matcher,config,device,match_threshold,ransac_threshold,min_distance));
        frame_log.open(output+"/frontend_frames.csv",std::ios::out);
        if (!frame_log) throw std::runtime_error("cannot open frontend_frames.csv in output_dir; create directory before launch");
        frame_log << "frame_id,stamp_ns,superpoint_calls,raw_keypoints,kept_keypoints,prev_track_count,prev_candidate_count,matcher_raw_pairs,matches_before_ransac,ransac_inliers_before_cap,matches_after_ransac,retained_tracks,published_points,extract_ms,match_ms,total_ms,receive_steady_ns,publish_steady_ns,generation\n";
        feature_pub=n.advertise<sensor_msgs::PointCloud>("feature",1000);
        ros::CallbackQueue reset_ack_queue;
        ros::NodeHandle reset_ack_node;
        reset_ack_node.setCallbackQueue(&reset_ack_queue);
        ros::Subscriber reset_ack_subscriber=reset_ack_node.subscribe(
            "/supergvins/reset_ack",10,resetAckCallback);
        reset_ack_subscriber_ptr=&reset_ack_subscriber;
        ros::AsyncSpinner reset_ack_spinner(1,&reset_ack_queue);
        reset_ack_spinner.start();
        auto subscriber=n.subscribe(topic,100,imageCallback);
        ROS_INFO_STREAM("SuperPoint + LightGlue ready; requested device="<<device<<", topic="<<topic
                        <<", candidate cache limit=512, active track limit=200, reset ack topic=/supergvins/reset_ack");
        ros::spin();
        subscriber.shutdown();
        reset_ack_subscriber.shutdown();
        reset_ack_subscriber_ptr=nullptr;
        reset_ack_spinner.stop();
        tracker.reset();
        frame_log.close();
    } catch (const std::exception& error) {
        ROS_FATAL_STREAM("frontend initialization or shutdown failed: "<<error.what());
        reset_ack_subscriber_ptr=nullptr;
        tracker.reset();
        if (frame_log.is_open()) frame_log.close();
        return 1;
    }
    return processing_failed ? 1 : 0;
}
