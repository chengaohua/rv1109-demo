
#include "firedet.h"
#include "tensor.h"

FireDet::FireDet() {}

FireDet::~FireDet() {

}

int FireDet::Init(const std::string path) {

    engine_.Init(path);
    return 0;
}

int FireDet::Process(cv::Mat & img, std::vector<cv::Rect> & rects) {

    cc::Tensor<u_int8_t> tensor;

    //todo preprocess
    tensor.from_cvmat(img, true);

std::vector<cc::Tensor<float>> outputTensors;
    engine_.forward(tensor, outputTensors);

    //postProcess;
    return 0;
    

    
}