//
// Created by gh on 2022/6/19.
//

#ifndef FIRE_DEMO_SRC_FACEFEAT_FACEFEAT_H_
#define FIRE_DEMO_SRC_FACEFEAT_FACEFEAT_H_

#include "../common/rknn_engine.h"
#include <string>

class FaceFeat {

 public:
  FaceFeat();
  ~FaceFeat();
  int Init(const std::string path);

  // use BGR
  int Process(cv::Mat & img, std::vector<cv::Point2f> points, std::vector<float> & feat);

  cc::RknnEngin engine_;

};


#endif //FIRE_DEMO_SRC_FACEFEAT_FACEFEAT_H_
