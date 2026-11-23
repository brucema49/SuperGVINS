cmake_minimum_required(VERSION 2.8.3)
project(supergvins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -DNDEBUG -Wall")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    rosbag
    gnss_comm
    gvins
)

find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    rosbag
    gnss_comm
    gvins
)

add_executable(${PROJECT_NAME}
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/factor/gnss_psr_dopp_factor.cpp
    src/factor/gnss_dt_ddt_factor.cpp
    src/factor/gnss_dt_anchor_factor.cpp
    src/factor/gnss_ddt_smooth_factor.cpp
    src/factor/pos_vel_factor.cpp
    src/factor/pose_anchor_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/gnss_spp_trace_writer.cpp
    src/utility/gnss_online_residual_trace_writer.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/initial/gnss_vi_initializer.cpp
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
add_dependencies(${PROJECT_NAME}
  ${catkin_EXPORTED_TARGETS}
)

if(CATKIN_ENABLE_TESTING)
  add_executable(supergvins_initial_sfm_geometry_test
    test/test_initial_sfm_geometry.cpp
    src/initial/initial_sfm.cpp
  )
  target_include_directories(supergvins_initial_sfm_geometry_test PRIVATE src)
  target_compile_options(supergvins_initial_sfm_geometry_test PRIVATE -UNDEBUG)
  target_link_libraries(supergvins_initial_sfm_geometry_test ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
  add_test(NAME supergvins_initial_sfm_geometry_test COMMAND supergvins_initial_sfm_geometry_test)

  catkin_add_gtest(supergvins_test_gnss_clock_prediction
    test/test_gnss_clock_prediction.cpp
  )
  if(TARGET supergvins_test_gnss_clock_prediction)
    target_include_directories(supergvins_test_gnss_clock_prediction PRIVATE src)
    target_link_libraries(supergvins_test_gnss_clock_prediction ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(supergvins_test_gnss_spp_trace_writer
    test/test_gnss_spp_trace_writer.cpp
    src/utility/gnss_spp_trace_writer.cpp
  )
  if(TARGET supergvins_test_gnss_spp_trace_writer)
    target_include_directories(supergvins_test_gnss_spp_trace_writer PRIVATE src)
    target_link_libraries(supergvins_test_gnss_spp_trace_writer ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(supergvins_test_gnss_online_residual_trace_writer
    test/test_gnss_online_residual_trace_writer.cpp
    src/factor/gnss_psr_dopp_factor.cpp
    src/utility/gnss_online_residual_trace_writer.cpp
  )
  if(TARGET supergvins_test_gnss_online_residual_trace_writer)
    target_include_directories(supergvins_test_gnss_online_residual_trace_writer PRIVATE src)
    target_link_libraries(supergvins_test_gnss_online_residual_trace_writer
      ${catkin_LIBRARIES}
      ${CERES_LIBRARIES}
    )
  endif()
endif()
