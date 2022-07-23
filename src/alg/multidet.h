#pragma once

#include "../common/rknn_engine.h"
#include "opencv2/opencv.hpp"

class MultiDet {

 public:
  MultiDet();
  ~MultiDet();
  int Init(const std::string ,float conf_thresh);

  int Process(cv::Mat &img, std::vector<cv::Rect> &rects, std::vector<float> &scores,std::vector<int> & cls);

  cc::RknnEngin engine_;
  float conf_thresh_;
};