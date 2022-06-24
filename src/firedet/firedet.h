#pragma once

#include "../common/rknn_engine.h"
#include "opencv2/opencv.hpp"

class FireDet {

 public:
  FireDet();
  ~FireDet();
  int Init(const std::string ,float conf_thresh);

  int Process(cv::Mat &img, std::vector<cv::Rect> &rects, std::vector<float> &scores);

  cc::RknnEngin engine_;
  float conf_thresh_;
};