# Minimum CMake required
cmake_minimum_required(VERSION 3.4)

# Project
project(fire-demo C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -fPIC -O2 ")

include_directories(3rdparty/include)
link_directories(3rdparty/libs)
link_directories(3rdparty/libs/opencv)
link_directories(3rdparty/libs/opencv/3rdparty/lib)

set(OPENCV_LIBS
        opencv_shape
        opencv_photo
        opencv_video
        opencv_videoio
        opencv_videostab
        opencv_features2d
        opencv_calib3d
        opencv_superres
        opencv_objdetect
        opencv_stitching
        opencv_imgcodecs
        opencv_imgproc
        opencv_highgui
        opencv_ml
        opencv_dnn
        opencv_flann
        opencv_core
        libpng
        libjpeg-turbo
        libtiff
        libjasper
        libprotobuf
        libwebp
        tegra_hal
        zlib
        )

add_executable(DEMO
 src/demo.cpp
 src/engine/rknn_engine.cpp
)

target_link_libraries(DEMO 
${OPENCV_LIBS}  rknn_api dl)
