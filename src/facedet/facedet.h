//
// Created by gh on 2022/6/19.
//

#ifndef FIRE_DEMO_SRC_FACEDET_FACEDET_H_
#define FIRE_DEMO_SRC_FACEDET_FACEDET_H_

#include "../common/rknn_engine.h"

class FaceDet {
 public:
  FaceDet();
  ~FaceDet();

  int Init(const std::string path);

  int Process(cv::Mat &img, std::vector<cv::Rect> &rects);

  cc::RknnEngin engine_;

 private:
  float thresh_ = 0.3;
};

#endif //FIRE_DEMO_SRC_FACEDET_FACEDET_H_
