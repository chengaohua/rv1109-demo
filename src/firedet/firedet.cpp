
#include "firedet.h"
#include "../tensor.h"
#include "postprocess.h"
#include "yoloV5.h"

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

    std::vector<cc::Tensor<float>> outputTensors;
    engine_.forward(tensor, outputTensors);

    // post process

    YoloV5 v5;
    auto boxes = v5.postProcess(outputTensors[0].data(), outputTensors[1].data(), outputTensors[2].data(),  img_width, img_height, 0.2);
    
    std::cout<<" box size"<<boxes.size()<<std::endl;

    for(int i = 0 ; i < boxes.size(); i++) {
      std::cout<<boxes[i].x1<<" "<<boxes[i].y1<<std::endl;
      std::cout<<boxes[i].x2<<" "<<boxes[i].y2<<std::endl;
    }
  
    // draw box
   
  

    //postProcess;
    return 0;
    

    
}