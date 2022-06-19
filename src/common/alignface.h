//
// Created by gh on 2022/6/19.
//

#ifndef FIRE_DEMO_SRC_COMMON_ALIGNFACE_H_
#define FIRE_DEMO_SRC_COMMON_ALIGNFACE_H_

cv::Mat alignFace(std::vector<cv::Point2f> &oriPoints, std::vector<cv::Point2f> & dstPoints, cv::Mat & img, int net_w, int net_h);

cv::Mat letterbox(cv::Mat & img, int net_w, int net_h);

#endif //FIRE_DEMO_SRC_COMMON_ALIGNFACE_H_
