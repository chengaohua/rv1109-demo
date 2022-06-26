
#include "firedet.h"
#include "../tensor.h"
#include "postprocess.h"
#include "yoloV5.h"
#include "../common/alignface.h"

FireDet::FireDet() {}

FireDet::~FireDet() {

}

int FireDet::Init(const std::string path, float thresh) {
  //std::cout<<"11box_conf_threshold = "<< conf_thresh_<<std::endl;
  conf_thresh_ = thresh;
  return engine_.Init(path);
}

int FireDet::Process(cv::Mat &img, std::vector<cv::Rect> &rects, std::vector<float> & scores, std::vector<int> & cls) {
  if(img.empty()) {
    return -1;
  }
  int width = 640;
  int height = 640;
  int img_width = img.cols;
  int img_height = img.rows;

  cv::Mat rsMat; // = letterbox(img, width, height);
  cv::resize(img,rsMat, cv::Size(width, height));
  cv::cvtColor(rsMat, rsMat, cv::COLOR_BGR2RGB);

  cc::Tensor<u_int8_t> tensor;

  //todo preprocess
  tensor.from_cvmat(rsMat, true);

  std::vector<cc::Tensor<uint8_t>> outputTensors;
  engine_.forward<uint8_t>(tensor, outputTensors);

  // post process
  float scale_w = (float) width / img_width;
  float scale_h = (float) height / img_height;

  float nms_threshold = NMS_THRESH;
  float box_conf_threshold = conf_thresh_;
  //std::cout<<"box_conf_threshold = "<< conf_thresh_<<std::endl;

  detect_result_group_t detect_result_group;
  std::vector<float> out_scales = {0.104080, 0.088612, 0.085162};
  std::vector<uint32_t> out_zps = {173, 167, 162};
  // for (int i = 0; i < io_num.n_output; ++i) {
  //   out_scales.push_back(output_attrs[i].scale);
  //   out_zps.push_back(output_attrs[i].zp);
  // }
  post_process((uint8_t *) outputTensors[0].data(),
               (uint8_t *) outputTensors[1].data(),
               (uint8_t *) outputTensors[2].data(),
               height,
               width,
               box_conf_threshold,
               nms_threshold,
               scale_w,
               scale_h,
               out_zps,
               out_scales,
               &detect_result_group);

  for (int i = 0; i < detect_result_group.count; i++) {
    detect_result_t *det_result = &(detect_result_group.results[i]);

    printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
           det_result->box.right, det_result->box.bottom, det_result->prop);
    int x1 = det_result->box.left;
    int y1 = det_result->box.top;
    int x2 = det_result->box.right;
    int y2 = det_result->box.bottom;
    
    cv::Rect rect(x1, y1, x2 -x1, y2 - y1);
    rects.push_back(rect);
    scores.push_back(det_result->prop);
    cls.push_back(det_result->cls);
    //rects.push_back(cv::Rect)
    #if 0
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2,y2),  cv::Scalar(0,255,0));
    #endif
    // draw box
  }

#if 0
  cv::imwrite("output.jpg", img);
#endif


  //postProcess;
  return 0;

}
