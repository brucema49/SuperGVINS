#include "learned_frontend.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace supergvins {
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

struct TensorContract {
    const char* name;
    ONNXTensorElementDataType type;
    std::vector<int64_t> shape;
};

std::string elementTypeName(ONNXTensorElementDataType type)
{
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    default: return "ONNX element type " + std::to_string(static_cast<int>(type));
    }
}

std::string shapeText(const std::vector<int64_t>& shape)
{
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) text << ',';
        if (shape[i] < 0) text << "dynamic";
        else text << shape[i];
    }
    text << ']';
    return text.str();
}

std::string namesText(const std::vector<std::string>& names)
{
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) text << ", ";
        text << names[i];
    }
    text << ']';
    return text.str();
}

void validateTensorList(const Ort::Session& session,
                        const char* direction,
                        const std::vector<TensorContract>& expected,
                        bool input)
{
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t actual_count = input ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> actual_names;
    actual_names.reserve(actual_count);
    for (size_t i = 0; i < actual_count; ++i) {
        auto allocated_name = input ? session.GetInputNameAllocated(i, allocator)
                                    : session.GetOutputNameAllocated(i, allocator);
        actual_names.emplace_back(allocated_name.get() ? allocated_name.get() : "");
    }

    std::vector<std::string> expected_names;
    expected_names.reserve(expected.size());
    std::map<std::string, const TensorContract*> contract_by_name;
    for (const TensorContract& contract : expected) {
        expected_names.emplace_back(contract.name);
        contract_by_name.emplace(contract.name, &contract);
    }
    if (actual_count != expected.size()) {
        std::ostringstream error;
        error << direction << " count/name mismatch: expected " << namesText(expected_names)
              << " but model declares " << namesText(actual_names);
        throw std::runtime_error(error.str());
    }

    for (size_t i = 0; i < actual_names.size(); ++i) {
        const auto contract = contract_by_name.find(actual_names[i]);
        if (contract == contract_by_name.end()) {
            std::ostringstream error;
            error << direction << " name mismatch: expected " << namesText(expected_names)
                  << " but model declares " << namesText(actual_names);
            throw std::runtime_error(error.str());
        }

        Ort::TypeInfo type_info = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
        if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
            throw std::runtime_error(std::string(direction) + " '" + actual_names[i] +
                                     "' must be a tensor");
        }
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType actual_type = tensor_info.GetElementType();
        const std::vector<int64_t> actual_shape = tensor_info.GetShape();
        const TensorContract& wanted = *contract->second;
        if (actual_type != wanted.type) {
            std::ostringstream error;
            error << direction << " '" << wanted.name << "' element type mismatch: expected "
                  << elementTypeName(wanted.type) << " but model declares "
                  << elementTypeName(actual_type);
            throw std::runtime_error(error.str());
        }
        if (actual_shape.size() != wanted.shape.size()) {
            std::ostringstream error;
            error << direction << " '" << wanted.name << "' rank mismatch: expected "
                  << shapeText(wanted.shape) << " but model declares " << shapeText(actual_shape);
            throw std::runtime_error(error.str());
        }
        for (size_t axis = 0; axis < wanted.shape.size(); ++axis) {
            const int64_t expected_dimension = wanted.shape[axis];
            const int64_t actual_dimension = actual_shape[axis];
            // -1 in the contract means runtime-variable, not "any dimension".
            if ((expected_dimension < 0 && actual_dimension >= 0) ||
                (expected_dimension >= 0 && actual_dimension != expected_dimension)) {
                std::ostringstream error;
                error << direction << " '" << wanted.name << "' shape mismatch: expected "
                      << shapeText(wanted.shape) << " but model declares " << shapeText(actual_shape);
                throw std::runtime_error(error.str());
            }
        }
    }
}

void validateModelSignature(const Ort::Session& session,
                            const std::string& role,
                            const std::string& path,
                            const std::vector<TensorContract>& expected_inputs,
                            const std::vector<TensorContract>& expected_outputs)
{
    try {
        validateTensorList(session, "input", expected_inputs, true);
        validateTensorList(session, "output", expected_outputs, false);
    } catch (const std::exception& error) {
        throw std::runtime_error(role + " model signature mismatch for '" + path + "': " + error.what());
    }
}

std::unique_ptr<Ort::Session> loadModelSession(Ort::Env& env,
                                               Ort::SessionOptions& options,
                                               const std::string& role,
                                               const std::string& path)
{
    try {
        return std::unique_ptr<Ort::Session>(new Ort::Session(env, path.c_str(), options));
    } catch (const std::exception& error) {
        throw std::runtime_error("failed to load " + role + " model '" + path + "': " + error.what());
    }
}
}
LearnedFrontend::LearnedFrontend(const std::string& extractor, const std::string& matcher,
                                 const std::string& camera, const std::string& device,
                                 float threshold, double ransac_threshold, int min_distance)
    : env_(ORT_LOGGING_LEVEL_WARNING, "supergvins"), threshold_(threshold),
      ransac_threshold_(ransac_threshold), min_distance_(min_distance)
{
    options_.SetIntraOpNumThreads(1);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (device == "cuda") {
        OrtCUDAProviderOptions cuda_options{};
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(options_, &cuda_options));
    } else if (device != "cpu") throw std::runtime_error("inference_device must be cpu or cuda");
    extractor_ = loadModelSession(env_, options_, "SuperPoint extractor", extractor);
    matcher_ = loadModelSession(env_, options_, "LightGlue matcher", matcher);
    validateModelSignature(
        *extractor_, "SuperPoint extractor", extractor,
        {{"image", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 1, -1, -1}}},
        {{"keypoints", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {1, -1, 2}},
         {"scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1}},
         {"descriptors", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1, 256}}});
    validateModelSignature(
        *matcher_, "LightGlue matcher", matcher,
        {{"kpts0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1, 2}},
         {"kpts1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1, 2}},
         {"desc0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1, 256}},
         {"desc1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1, 256}}},
        {{"matches0", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1, 2}},
         {"mscores0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1}}});
    camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(camera);
    check(static_cast<bool>(camera_), "camera calibration could not be loaded");
}
void LearnedFrontend::reset() { previous_candidates_.clear(); previous_time_ = 0; }
cv::Point2f LearnedFrontend::normalized(const cv::Point2f& pixel) const
{
    Eigen::Vector3d ray;
    camera_->liftProjective(Eigen::Vector2d(pixel.x,pixel.y),ray);
    check(std::isfinite(ray.z()) && std::abs(ray.z()) > 1e-12, "invalid camera ray");
    cv::Point2f p(ray.x()/ray.z(),ray.y()/ray.z());
    check(std::isfinite(p.x) && std::isfinite(p.y), "non-finite camera coordinates");
    return p;
}
std::vector<Feature> LearnedFrontend::extract(const cv::Mat& gray, FrameStats& stats)
{
    const float scale = 512.0f / std::max(gray.cols, gray.rows);
    cv::Mat resized, input;
    cv::resize(gray,resized,cv::Size(std::lround(gray.cols*scale),std::lround(gray.rows*scale)),0,0,cv::INTER_AREA);
    resized.convertTo(input,CV_32F,1.0/255.0);
    check(input.isContinuous(), "non-contiguous input image");
    std::array<int64_t,4> shape{{1,1,input.rows,input.cols}};
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto tensor=Ort::Value::CreateTensor<float>(mem,input.ptr<float>(),input.total(),shape.data(),shape.size());
    const char* inputs[]={"image"};
    const char* outputs[]={"keypoints","scores","descriptors"};
    ++stats.extractor_calls;
    auto out=extractor_->Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,3);
    const auto ki=out[0].GetTensorTypeAndShapeInfo();
    const auto si=out[1].GetTensorTypeAndShapeInfo();
    const auto di=out[2].GetTensorTypeAndShapeInfo();
    const auto ks=ki.GetShape(), ds=di.GetShape();
    check(ki.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && si.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && di.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"invalid SuperPoint output types");
    check(ks.size()==3 && ks[0]==1 && ks[2]==2 && ks[1]>=0 && ds.size()==3 && ds[0]==1 && ds[1]==ks[1] && ds[2]==256,"invalid SuperPoint output shapes");
    const size_t n=static_cast<size_t>(ks[1]);
    check(si.GetElementCount()==n && di.GetElementCount()==n*256,"inconsistent SuperPoint output counts");
    stats.raw=n;
    if (n==0) return {};
    const auto* points=out[0].GetTensorData<int64_t>();
    const auto* scores=out[1].GetTensorData<float>();
    const auto* descriptors=out[2].GetTensorData<float>();
    std::vector<float> score_values(scores,scores+n);
    auto indices=selectKeypoints(score_values,512);
    std::vector<Feature> features;
    features.reserve(indices.size());
    for (size_t i:indices) {
        Feature f;
        f.model=cv::Point2f(points[i*2],points[i*2+1]);
        f.pixel=cv::Point2f((f.model.x+0.5f)/scale-0.5f,(f.model.y+0.5f)/scale-0.5f);
        if (f.pixel.x<1 || f.pixel.y<1 || f.pixel.x>=gray.cols-1 || f.pixel.y>=gray.rows-1) continue;
        f.normalized=normalized(f.pixel);
        std::copy(descriptors+i*256,descriptors+(i+1)*256,f.descriptor.begin());
        for (float value:f.descriptor) check(std::isfinite(value), "non-finite SuperPoint descriptor");
        features.push_back(f);
    }
    stats.candidates=features.size();
    return features;
}
std::vector<Match> LearnedFrontend::match(const std::vector<Feature>& current, int width, int height, FrameStats& stats)
{
    if (previous_candidates_.empty() || current.empty()) return {};
    const size_t n0=std::max<size_t>(128,previous_candidates_.size());
    const size_t n1=std::max<size_t>(128,current.size());
    check(previous_candidates_.size()<=kMaxCandidateCount && current.size()<=kMaxCandidateCount,
          "LightGlue candidates exceed the 512 point budget");
    std::vector<float> k0(n0*2,-10.f), k1(n1*2,-10.f), d0(n0*256,0.f), d1(n1*256,0.f);
    const float half=std::max(width,height)/2.f;
    auto fill=[&](const std::vector<Feature>& features,std::vector<float>& keypoints,std::vector<float>& desc) {
        for (size_t i=0;i<features.size();++i) {
            keypoints[2*i]=(features[i].model.x-width/2.f)/half;
            keypoints[2*i+1]=(features[i].model.y-height/2.f)/half;
            std::copy(features[i].descriptor.begin(),features[i].descriptor.end(),desc.begin()+i*256);
        }
    };
    fill(previous_candidates_,k0,d0); fill(current,k1,d1);
    std::array<int64_t,3> kshape0{{1,static_cast<int64_t>(n0),2}}, kshape1{{1,static_cast<int64_t>(n1),2}},
        dshape0{{1,static_cast<int64_t>(n0),256}}, dshape1{{1,static_cast<int64_t>(n1),256}};
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::vector<Ort::Value> in;
    in.push_back(Ort::Value::CreateTensor<float>(mem,k0.data(),k0.size(),kshape0.data(),3));
    in.push_back(Ort::Value::CreateTensor<float>(mem,k1.data(),k1.size(),kshape1.data(),3));
    in.push_back(Ort::Value::CreateTensor<float>(mem,d0.data(),d0.size(),dshape0.data(),3));
    in.push_back(Ort::Value::CreateTensor<float>(mem,d1.data(),d1.size(),dshape1.data(),3));
    const char* inputs[]={"kpts0","kpts1","desc0","desc1"};
    const char* outputs[]={"matches0","mscores0"};
    auto out=matcher_->Run(Ort::RunOptions{nullptr},inputs,in.data(),in.size(),outputs,2);
    auto mi=out[0].GetTensorTypeAndShapeInfo(), si=out[1].GetTensorTypeAndShapeInfo();
    auto ms=mi.GetShape(), ss=si.GetShape();
    check(mi.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && si.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"invalid LightGlue output types");
    check(ms.size()==2 && ms[1]==2 && ms[0]>=0 && ss.size()==1 && ss[0]==ms[0],"invalid LightGlue output shapes");
    const auto* pairs=out[0].GetTensorData<int64_t>();
    const auto* scores=out[1].GetTensorData<float>();
    std::vector<Match> candidates;
    stats.raw_matches=static_cast<size_t>(ms[0]);
    for (int64_t i=0;i<ms[0];++i) {
        check(pairs[2*i]>=0 && pairs[2*i]<static_cast<int64_t>(n0) && pairs[2*i+1]>=0 && pairs[2*i+1]<static_cast<int64_t>(n1),"LightGlue returned out-of-range index");
        candidates.push_back({static_cast<int>(pairs[2*i]),static_cast<int>(pairs[2*i+1]),scores[i]});
    }
    auto result=selectMatches(candidates,previous_candidates_.size(),current.size(),threshold_,kMaxCandidateCount);
    stats.matches=result.size();
    if (result.size()>=8) {
        std::vector<cv::Point2f> a,b;
        for (const auto& m:result) {
            auto p=previous_candidates_[m.previous].normalized, q=current[m.current].normalized;
            a.emplace_back(460*p.x+width/2.f,460*p.y+height/2.f);
            b.emplace_back(460*q.x+width/2.f,460*q.y+height/2.f);
        }
        std::vector<uchar> mask;
        cv::Mat f=cv::findFundamentalMat(a,b,cv::FM_RANSAC,ransac_threshold_,0.99,mask);
        if (f.empty() || mask.size()!=result.size()) result.clear();
        else {
            std::vector<Match> filtered;
            for (size_t i=0;i<result.size();++i) if (mask[i]) filtered.push_back(result[i]);
            result.swap(filtered);
        }
    }
    stats.ransac_inliers_before_cap=result.size();
    std::vector<int> previous_track_ages;
    previous_track_ages.reserve(previous_candidates_.size());
    for (const auto& feature : previous_candidates_)
        previous_track_ages.push_back(feature.id >= 0 ? feature.age : 0);
    capMatchesByTrackAge(result,kMaxActiveTrackCount,previous_track_ages);
    stats.inliers=result.size();
    return result;
}
std::vector<Feature> LearnedFrontend::process(const cv::Mat& gray, double timestamp, FrameStats& stats)
{
    stats=FrameStats{};
    check(!gray.empty() && gray.type()==CV_8UC1,"expected non-empty mono8 image");
    check(std::isfinite(timestamp) && (previous_time_==0 || timestamp>previous_time_),"non-increasing image timestamp");
    const auto started=std::chrono::steady_clock::now();
    stats.previous_candidates=previous_candidates_.size();
    for (const Feature& feature:previous_candidates_)
        if (feature.id>=0) ++stats.previous;
    check(previous_candidates_.size()<=kMaxCandidateCount,
          "cached LightGlue candidates exceed the 512 point budget");
    auto candidates=extract(gray,stats);
    const auto extracted=std::chrono::steady_clock::now();
    auto matched=match(candidates,gray.cols,gray.rows,stats);
    const auto matched_at=std::chrono::steady_clock::now();
    stats.extract_ms=std::chrono::duration<double,std::milli>(extracted-started).count();
    stats.match_ms=std::chrono::duration<double,std::milli>(matched_at-extracted).count();
    std::vector<Feature> tracks;
    tracks.reserve(std::min(kMaxActiveTrackCount,candidates.size()));
    std::vector<bool> used(candidates.size());
    cv::Mat mask(gray.size(),CV_8UC1,cv::Scalar(255));
    for (const auto& m:matched) {
        check(m.previous>=0 && static_cast<size_t>(m.previous)<previous_candidates_.size() &&
              m.current>=0 && static_cast<size_t>(m.current)<candidates.size(),
              "selected LightGlue match index became invalid");
        Feature f=candidates[m.current]; const auto& prev=previous_candidates_[m.previous];
        const MatchedTrackState state=advanceMatchedTrack(prev.id,prev.age,
            prev.normalized.x,prev.normalized.y,f.normalized.x,f.normalized.y,
            timestamp-previous_time_,next_id_);
        f.id=state.id; f.age=state.age;
        f.velocity=cv::Point2f(state.velocity_x,state.velocity_y);
        check(used[m.current]==false,"multiple selected matches target the same current feature");
        tracks.push_back(f); used[m.current]=true;
        candidates[m.current]=f;
        cv::circle(mask,f.pixel,min_distance_,cv::Scalar(0),-1);
    }
    for (size_t i=0;i<candidates.size() && tracks.size()<kMaxActiveTrackCount;++i) {
        if (used[i]) continue;
        const auto& p=candidates[i].pixel;
        const int x=cvRound(p.x),y=cvRound(p.y);
        if (x<0 || y<0 || x>=gray.cols || y>=gray.rows || mask.at<uchar>(y,x)==0) continue;
        check(next_id_ >= 0 && next_id_ < (1<<24),"feature id exhausted float32 precision; coordinated restart required");
        Feature f=candidates[i]; f.id=next_id_++; f.velocity=cv::Point2f(0,0);
        f.age=1;
        tracks.push_back(f); candidates[i]=f;
        cv::circle(mask,p,min_distance_,cv::Scalar(0),-1);
    }
    std::set<int> current_ids;
    for (const Feature& feature:tracks)
        check(feature.id>=0 && current_ids.insert(feature.id).second,
              "duplicate or invalid active feature id after track update");
    check(tracks.size()<=kMaxActiveTrackCount,"active track set exceeds the 200 point budget");
    check(candidates.size()<=kMaxCandidateCount,"candidate cache exceeds the 512 point budget");
    previous_candidates_=std::move(candidates);
    previous_time_=timestamp; stats.retained=tracks.size();
    stats.total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    return tracks;
}
}
