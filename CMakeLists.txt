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
        ittnotify
        zlib
        )


add_library(cc SHARED src/common/rknn_engine.cpp
        src/firedet/firedet.cpp
        src/interface/interface.cpp
        src/firedet/postprocess.cc
        src/firedet/yoloV5.cpp
        src/common/alignface.cpp)


target_link_libraries(cc
${OPENCV_LIBS}  rknn_api dl pthread)

add_executable(DEMO src/demo.cpp)
target_link_libraries(DEMO
        cc ${OPENCV_LIBS})


add_executable(DEMO2
        src/yolov5demo.cpp
       )

target_link_libraries(DEMO2
        ${OPENCV_LIBS}  rknn_api dl pthread)