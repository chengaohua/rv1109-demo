#pragma once

#include "../engine/rknn_engine.h"
#include "opencv2/opencv.hpp"

class FireDet {

public:
FireDet();
~FireDet();
int Init(const std::string path);

int Process(cv::Mat & img, std::vector<cv::Rect> & rects);

cc::RknnEngin engine_;

};