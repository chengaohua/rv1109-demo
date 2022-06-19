//
// Created by gh on 2022/6/19.
//

#include "facedet.h"
#include "../common/alignface.h"

#define IMO_MAX(x, y) (x)>(y)?(x):(y)
#define IMO_MIN(x, y) (x) < (y) ? (x) : (y)

const float MAX_HEIGHT = 160.0;
const float MAX_WIDTH = 160.0;

const int net_width = 160;
const int net_height = 160;

using ImageWH = struct {
  float width;
  float height;
};

ImageWH calacOutputWH(ImageWH input) {

  ImageWH output{0, 0};
  if (input.width == 0 || input.height == 0) {
    return output;
  }

  float s = IMO_MIN(MAX_WIDTH / input.width, MAX_HEIGHT / input.height);

  //四舍五入
  int nw = (s * input.width + 0.5);
  int nh = (s * input.height + 0.5);

  output.width = nw;
  output.height = nh;

  return output;
};



FaceDet::FaceDet() {

}

FaceDet::~FaceDet() {

}

int FaceDet::Init(const std::string path) {
  engine_.Init(path);
  return 0;
}

int FaceDet::Process(cv::Mat &img, std::vector<cv::Rect> &rects) {

  ImageWH inputWH;
  inputWH.width = img.cols;
  inputWH.height = img.rows;

  ImageWH outputWH = calacOutputWH(inputWH);
  int half_offset_x = (MAX_WIDTH - outputWH.width) / 2;
  int half_offset_y = (MAX_HEIGHT - outputWH.height) / 2;


  cv::Mat rsMat = letterbox(img, net_width, net_height);

  cc::Tensor<u_int8_t> tensor;

  //todo preprocess
  tensor.from_cvmat(rsMat, true);

  std::vector<cc::Tensor<float>> outputTensors;
  engine_.forward(tensor, outputTensors);







  //todo change idx
  float * hmFlat = outputTensors[0].data();
  float * whFlat = outputTensors[1].data();
  float * regFlat = outputTensors[2].data();

  std::vector<int> inds;
  std::vector<std::pair<float, float>> centors;

  const int out_h = MAX_HEIGHT / 4;
  const int out_w = MAX_WIDTH / 4;

  auto len = out_h * out_w;

  const int chan = 1;
  //查找满足条件的中心点坐标
  for (int i = 0; i < out_h * chan; i++) {
    for (int j = 0; j < out_w; j++) {
      auto id = i * out_w + j;
      float tmp = hmFlat[id] == hmFlat[id + len] ? hmFlat[id] : 0;
      if (tmp > thresh_) {
        inds.push_back(id);
        centors.push_back(std::make_pair(j, i));
      }
    }
  }

  std::cout << inds.size() << std::endl;
  std::vector<float> scores;
  std::vector<std::pair<float, float>> wh;
  std::vector<std::pair<float, float>> reg;
  //std::vector<std::vector<float>> kps;

  for (auto idx : inds) {
    auto score = hmFlat[idx];

    scores.push_back(score);
    auto wh1 = whFlat[idx];
    auto wh2 = whFlat[idx + len];

    auto reg1 = regFlat[idx];
    auto reg2 = regFlat[idx + len];

    wh.push_back(std::make_pair(wh1, wh2));

    reg.push_back(std::make_pair(reg1, reg2));


//            std::vector<float> landmarks;
//            landmarks.push_back(kpsFlat[idx]);
//            landmarks.push_back(kpsFlat[  idx + 1 * len]);
//            landmarks.push_back(kpsFlat[  idx + 2 * len]);
//            landmarks.push_back(kpsFlat[  idx + 3 * len]);
//            landmarks.push_back(kpsFlat[  idx + 4* len]);
//            landmarks.push_back(kpsFlat[  idx + 5* len]);
//            landmarks.push_back(kpsFlat[  idx + 6* len]);
//            landmarks.push_back(kpsFlat[  idx + 7* len]);
//            landmarks.push_back(kpsFlat[  idx + 8* len]);
//            landmarks.push_back(kpsFlat[  idx + 9* len]);
//            kps.push_back(landmarks);
  }

  const int rect_num = inds.size();

  float x_ratio = (float)  inputWH.width/ outputWH.width;
  float y_ratio = (float) inputWH.height / outputWH.height;

  //矫正landmarks
//        for (int i = 0; i < rect_num; i++) {
//            kps[i][0] += centors[i].first;
//            kps[i][1] += centors[i].second;
//            kps[i][2] += centors[i].first;
//            kps[i][3] += centors[i].second;
//            kps[i][4] += centors[i].first;
//            kps[i][5] += centors[i].second;
//            kps[i][6] += centors[i].first;
//            kps[i][7] += centors[i].second;
//            kps[i][8] += centors[i].first;
//            kps[i][9] += centors[i].second;
//
//        }

  //矫正centor
  for (int i = 0; i < rect_num; i++) {
    centors[i].first += reg[i].first;
    centors[i].second += reg[i].second;

    //  cv::Rect2f rect;
    float x = centors[i].first - wh[i].first / 2;
    float y = centors[i].second - wh[i].second / 2;
    float width = wh[i].first;
    float height = wh[i].second;

    //映射到原始图
    x = (4 * x - half_offset_x) * x_ratio;
    y = (4 * y - half_offset_y) * y_ratio;

    width *= (4 * x_ratio);
    height *= (4 * y_ratio);

    x = std::max(std::min(inputWH.width, (int) x), 0);
    y = std::max(std::min(inputWH.height, (int) y), 0);

    width = std::max(std::min((float) (inputWH.width - x), width), 1.0f);
    height = std::max(std::min((float) (inputWH.height - y), height), 1.0f);

    rects.emplace_back(cv::Rect{(int)x, (int)y, (int)width, (int)height});
  }






}