cmake_minimum_required(VERSION 2.8.3)
project(supergvins_feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -DNDEBUG -Wall")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    std_msgs
    sensor_msgs
    cv_bridge
    gvins_camera_model
    )

find_package(OpenCV REQUIRED)

if(NOT ONNXRUNTIME_ROOTDIR)
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOTDIR to the directory containing ONNX Runtime")
endif()
find_library(SUPERGVINS_ONNXRUNTIME_LIBRARY NAMES onnxruntime libonnxruntime.so.1.23.2
  PATHS "${ONNXRUNTIME_ROOTDIR}" "${ONNXRUNTIME_ROOTDIR}/lib" NO_DEFAULT_PATH)
if(NOT SUPERGVINS_ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime library not found in ONNXRUNTIME_ROOTDIR=${ONNXRUNTIME_ROOTDIR}")
endif()
get_filename_component(SUPERGVINS_ORT_DIR "${SUPERGVINS_ONNXRUNTIME_LIBRARY}" DIRECTORY)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(supergvins_feature_tracker
    src/feature_tracker_node.cpp
    src/learned_frontend.cpp
    )

target_link_libraries(supergvins_feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS} ${SUPERGVINS_ONNXRUNTIME_LIBRARY})
set_target_properties(supergvins_feature_tracker PROPERTIES BUILD_RPATH "${SUPERGVINS_ORT_DIR}" INSTALL_RPATH "${SUPERGVINS_ORT_DIR}")
if(CATKIN_ENABLE_TESTING)
  add_executable(supergvins_frontend_policy_test test/frontend_policy_test.cpp)
  target_include_directories(supergvins_frontend_policy_test PRIVATE src)
  target_compile_options(supergvins_frontend_policy_test PRIVATE -UNDEBUG)
  add_test(NAME supergvins_frontend_policy_test COMMAND supergvins_frontend_policy_test)

  add_executable(supergvins_track_continuity_test test/track_continuity_test.cpp)
  target_include_directories(supergvins_track_continuity_test PRIVATE src)
  target_compile_options(supergvins_track_continuity_test PRIVATE -UNDEBUG)
  add_test(NAME supergvins_track_continuity_test COMMAND supergvins_track_continuity_test)

  add_executable(supergvins_frontend_signature_test
    test/learned_frontend_signature_test.cpp
    src/learned_frontend.cpp
    )
  target_include_directories(supergvins_frontend_signature_test PRIVATE src)
  target_link_libraries(supergvins_frontend_signature_test
    ${catkin_LIBRARIES} ${OpenCV_LIBS} ${SUPERGVINS_ONNXRUNTIME_LIBRARY})
  set_target_properties(supergvins_frontend_signature_test PROPERTIES
    BUILD_RPATH "${SUPERGVINS_ORT_DIR}" INSTALL_RPATH "${SUPERGVINS_ORT_DIR}")
  add_test(NAME supergvins_frontend_signature_test
    COMMAND supergvins_frontend_signature_test
      ${PROJECT_SOURCE_DIR}/../weights/superpoint.onnx
      ${PROJECT_SOURCE_DIR}/../weights/superpoint_lightglue_fused_cpu.onnx
      ${PROJECT_SOURCE_DIR}/../config/visensor_f9p/visensor_left_f9p_config.yaml)
endif()
