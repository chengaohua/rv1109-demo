//
// Created by gh on 2022/6/19.
//

#include "interface.h"
#include "../alg/firedet.h"
#include "../common/alignface.h"

int fire_det_create(cc_det_handle *handle, const char *model, float thresh) {
  if (handle == nullptr || model == nullptr) {
    return -1;
  }

  FireDet *fire_det = new FireDet;

  std::string path = model;

  //std::cout<<"box_conf_threshold = "<< thresh<<std::endl;
  auto ret = fire_det->Init(model, thresh);

  if (ret < 0) {
    delete fire_det;
  }

  handle->handle = fire_det;

  return 0;
}

//now get top 10 , size is rect num
int fire_det_exec(const cc_det_handle *handle, cc_image *img, cc_rect rect[10], int * size) {
  FireDet *fire_det = (FireDet *) (handle->handle);

  cv::Mat bgrMat;
  auto ret = CCImage2BgrMat(img, &bgrMat);
  if (ret < 0) {
    return -1;
  }
  std::vector<cv::Rect> rects;
  std::vector<float>  scores;
  std::vector<int> cls;
  fire_det->Process(bgrMat, rects, scores, cls);

  auto len = rects.size() > 10 ? 10 : rects.size() ;
  for (int i = 0; i < len; i++) {
    rect[i].x = rects[i].x;
    rect[i].y = rects[i].y;
    rect[i].width = rects[i].width;
    rect[i].height = rects[i].height;
    rect[i].conf = scores[i];
    rect[i].cls = cls[i];

  }

  *size = len;
  return 0;
}

int fire_det_destroy(cc_det_handle *handle){
  FireDet *fire_det = (FireDet *) (handle->handle);
  delete fire_det;
  handle->handle = nullptr;
  return 0;
}


#include "../alg/stationerydet.h"


int stationery_det_create(cc_det_handle *handle, const char *model, float thresh) {
  if (handle == nullptr || model == nullptr) {
    return -1;
  }

  StationeryDet *sta_det = new StationeryDet;

  std::string path = model;

  //std::cout<<"box_conf_threshold = "<< thresh<<std::endl;
  auto ret = sta_det->Init(model, thresh);

  if (ret < 0) {
    delete sta_det;
  }

  handle->handle = sta_det;

  return 0;
}

//now get top 100 , size is rect num
int stationery_det_exec(const cc_det_handle *handle, cc_image *img, cc_rect rect[100], int * size) {
  StationeryDet *sta_det = (StationeryDet *) (handle->handle);

  cv::Mat bgrMat;
  auto ret = CCImage2BgrMat(img, &bgrMat);
  if (ret < 0) {
    return -1;
  }
  std::vector<cv::Rect> rects;
  std::vector<float>  scores;
  std::vector<int> cls;
  sta_det->Process(bgrMat, rects, scores, cls);

  auto len = rects.size() > 100 ? 100 : rects.size() ;
  for (int i = 0; i < len; i++) {
    rect[i].x = rects[i].x;
    rect[i].y = rects[i].y;
    rect[i].width = rects[i].width;
    rect[i].height = rects[i].height;
    rect[i].conf = scores[i];
    rect[i].cls = cls[i];

  }

  *size = len;
  return 0;
}

int stationery_det_destroy(cc_det_handle *handle){
 StationeryDet *sta_det = (StationeryDet *) (handle->handle);
 delete sta_det;
 handle->handle = nullptr;
  return 0;
}


