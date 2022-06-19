//
// Created by gh on 2022/6/19.
//

#include "faceFeat.h"
#include "../common/alignface.h"

FaceFeat::FaceFeat() {

}

FaceFeat::~FaceFeat() {

}

int FaceFeat::Init(const std::string path) {
  engine_.Init(path);
  return 0;
}

int FaceFeat::Process(cv::Mat &img, std::vector<cv::Point2f> points, std::vector<float> &feat) {
  const int net_w = 112;
  const int net_h = 112;

  std::vector<cv::Point2f> dstPoints = {{37.41309091, 43.78181818},
                                        {75.58676364, 43.78181818},
                                        {56.44026182, 86.05061818}};

  auto alignMat = alignFace(points, dstPoints, img, net_w, net_h);

  // (img - 127.5f) * 3.2/255;
//  const float meanVals[3] = {127.5f, 127.5f, 127.5f};
//  const float normVals[3] = {3.2f / 255.0f, 3.2f / 255.0f, 3.2f / 255.0f};
//
//  std::shared_ptr<MNN::CV::ImageProcess> pretreat(
//      MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, meanVals, 3, normVals, 3)
//  );

  cc::Tensor<u_int8_t> tensor;

  //todo preprocess
  tensor.from_cvmat(alignMat, true);

  std::vector<cc::Tensor<float>> outputTensors;
  engine_.forward(tensor, outputTensors);

  feat.resize(outputTensors[0].size());
  memcpy(feat.data(), outputTensors[0].data(), outputTensors[0].size() * sizeof(float));

  return 0;
}

