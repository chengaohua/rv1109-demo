
#include "firedet.h"
#include "../tensor.h"
#include "postprocess.h"
#include "yoloV5.h"
#include "../common/alignface.h"

FireDet::FireDet() {}

FireDet::~FireDet() {

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