//
// Created by gh on 2022/6/19.
//

#ifndef FIRE_DEMO_SRC_PERSON_DET__H_
#define FIRE_DEMO_SRC_PERSON_DET__H_

#include "../common/rknn_engine.h"
#include <string>

class PersonDet {

 public:
  PersonDet();
  ~PersonDet();
  int Init(const std::string path);

  // use BGR
  int Process(cv::Mat & img, std::vector<cv::Rect> &rects);

  cc::RknnEngin engine_;

};


#endif //FIRE_DEMO_SRC_PERSON_DET__H_
