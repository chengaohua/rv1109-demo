
#include "firedet.h"
#include "../tensor.h"
#include "postprocess.h"

FireDet::FireDet() {}

FireDet::~FireDet() {

}

cv::Mat letterbox(cv::Mat & img, int net_w, int net_h) {
  //auto &img = p->image_;
  const auto input_w = net_w;
  const auto input_h = net_h;
  const auto borderValue = 114;

  cv::Mat resized;

  if (input_w == input_h) {
    int dest_w(input_w), dest_h(input_h);
    int padbottom(0), padright(0);
    int padtop = 0;
    int padleft = 0;
    if (img.cols > img.rows) {
      dest_h = static_cast<int>(img.rows * input_h / static_cast<float>(img.cols));
      padbottom = input_h - dest_h;
      padbottom = padbottom >= 0 ? padbottom : 0;
    } else {
      dest_w = static_cast<int>(img.cols * input_w / static_cast<float>(img.rows));
      padright = input_w - dest_w;
      padright = padright >= 0 ? padright : 0;
    }

    // 居中对齐
    {
      padtop = padbottom / 2;
      padleft = padright / 2;
      padbottom -= padtop;
      padright -= padleft;
    }

    cv::resize(img, resized, cv::Size(dest_w, dest_h), 0.0f, 0.0f);
    cv::copyMakeBorder(resized,
                       resized,
                       padtop,
                       padbottom,
                       padleft,
                       padright,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(borderValue, borderValue, borderValue));
  } else {
    cv::resize(img, resized, cv::Size(input_w, input_h));
  }

  return resized;
}

int FireDet::Init(const std::string path) {
    engine_.Init(path);
    return 0;
}

int FireDet::Process(cv::Mat & img, std::vector<cv::Rect> & rects) {

    int width = 640;
    int height = 640;
    int img_width= img.cols;
    int img_height = img.rows;

    cv::Mat rsMat = letterbox(img, width, height);
    //todo
    //cv::cvtColor

    cv::cvtColor(rsMat, rsMat, cv::COLOR_BGR2RGB);

    cc::Tensor<u_int8_t> tensor;

    //todo preprocess
    tensor.from_cvmat(rsMat, true);

    std::vector<cc::Tensor<uint8_t>> outputTensors;
    engine_.forward(tensor, outputTensors);

    // post process
  float scale_w = (float)width / img_width;
  float scale_h = (float)height / img_height;

  const float    nms_threshold      = NMS_THRESH;
  const float    box_conf_threshold = BOX_THRESH;

  detect_result_group_t detect_result_group;
  std::vector<float>    out_scales = {0.104080, 0.088612, 0.085162};
  std::vector<uint32_t> out_zps = {173,167,162};
  // for (int i = 0; i < io_num.n_output; ++i) {
  //   out_scales.push_back(output_attrs[i].scale);
  //   out_zps.push_back(output_attrs[i].zp);
  // }
  post_process((uint8_t*)outputTensors[0].data(), (uint8_t*)outputTensors[1].data(), (uint8_t*)outputTensors[2].data(), height, width,
               box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

               for (int i = 0; i < detect_result_group.count; i++) {
    detect_result_t* det_result = &(detect_result_group.results[i]);
   
    printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
           det_result->box.right, det_result->box.bottom, det_result->prop);
    int x1 = det_result->box.left;
    int y1 = det_result->box.top;
    int x2 = det_result->box.right;
    int y2 = det_result->box.bottom;
    // draw box
   
  }

    //postProcess;
    return 0;
    

    
}