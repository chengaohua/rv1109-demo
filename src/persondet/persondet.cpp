
#include "persondet.h"
#include "../tensor.h"
#include "../common/alignface.h"
#include "Yolo.h"

PersonDet::PersonDet() {}

PersonDet::~PersonDet() {

}

int PersonDet::Init(const std::string path, float thresh) {
  //std::cout<<"11box_conf_threshold = "<< conf_thresh_<<std::endl;
  conf_thresh_ = thresh;
  return engine_.Init(path);
}

int PersonDet::Process(cv::Mat &img, std::vector<cv::Rect> &rects, std::vector<float> & scores, std::vector<int> & cls) {
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

  std::vector<cc::Tensor<float>> outputTensors;
  engine_.forward<float>(tensor, outputTensors);


  int num_classes=5;
    std::vector<YoloLayerData> yolov5ss_layers{
      {"output", 8,  {{10,  13}, {16,  30},  {33,  23}}},
      {"395",    16, {{30,  61}, {62,  45},  {59,  119}}},
      {"415",    32, {{116, 90}, {156, 198}, {373, 326}}},
          
            
    };
    std::vector<YoloLayerData> & layers = yolov5ss_layers;
    std::vector<std::string> labels{"person", "vehicle", "outdoor", "animal", "accessory"};


  std::vector<BoxInfo> result;
  std::vector<BoxInfo> boxes;
    
  yolocv::YoloSize yolosize = yolocv::YoloSize{640,640};
    
  float threshold = 0.3;
  float nms_threshold = 0.7;

  boxes = decode_infer(outputTensors[0], layers[0].stride,  yolosize, 640, num_classes, layers[0].anchors, threshold);
  result.insert(result.begin(), boxes.begin(), boxes.end());

  boxes = decode_infer(outputTensors[1], layers[1].stride,  yolosize, 640, num_classes, layers[1].anchors, threshold);
  result.insert(result.begin(), boxes.begin(), boxes.end());

  boxes = decode_infer(outputTensors[2], layers[2].stride,  yolosize, 640, num_classes, layers[2].anchors, threshold);
  result.insert(result.begin(), boxes.begin(), boxes.end());

  nms(result, nms_threshold);

  std::cout<<result.size()<<std::endl;


  //std::cout<<"---------------->"<<outputTensors[0].shape()<<std::endl;
  // int batchs, channels, height, width, pred_item ;
  // batchs = data.shape()[0];
  // channels = data.shape()[1];
  // height = data.shape()[2];
  // width = data.shape()[3];
  // pred_item = data.shape()[4];

  

  //postProcess;
  return 0;

}
