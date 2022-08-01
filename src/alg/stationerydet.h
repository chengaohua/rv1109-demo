#pragma once

#include "../common/rknn_engine.h"
#include "opencv2/opencv.hpp"

class StationeryDet {

 public:
  StationeryDet();
  ~StationeryDet();
  int Init(const std::string ,float conf_thresh);

  int Process(cv::Mat &img, std::vector<cv::Rect> &rects, std::vector<float> &scores,std::vector<int> & cls);

  cc::RknnEngin engine_;
  float conf_thresh_ = 0.25;
  float nms_thresh_ = 0.45;
};