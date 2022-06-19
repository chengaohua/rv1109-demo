
#include "yoloV5.h"

#include <math.h>
#include <iostream>
#include <algorithm>
#include <functional>

std::vector<int> YoloV5::get_anchors(int net_grid) {
  std::vector<int> anchors(6);
  int a80[6] = {10, 13, 16, 30, 33, 23};
  int a40[6] = {30, 61, 62, 45, 59, 119};
  int a20[6] = {116, 90, 156, 198, 373, 326};
  if (net_grid == 80) {
    anchors.insert(anchors.begin(), a80, a80 + 6);
  } else if (net_grid == 40) {
    anchors.insert(anchors.begin(), a40, a40 + 6);
  } else if (net_grid == 20) {
    anchors.insert(anchors.begin(), a20, a20 + 6);
  }
  return anchors;
}

void YoloV5::doNMS(std::vector<Bbox> &bboxes, float nms_thresh) {
  std::cout <<"thresh:"<<nms_thresh<<std::endl;
  std::sort(bboxes.begin(), bboxes.end(), [](Bbox a, Bbox b) {
    return a.score > b.score;
  });
  std::vector<float> vArea(bboxes.size());
  for (int i = 0; i < int(bboxes.size()); ++i) {
    vArea[i] = (bboxes.at(i).x2 - bboxes.at(i).x1 + 1) * (bboxes.at(i).y2 - bboxes.at(i).y1 + 1);

  }
  for (int i = 0; i < int(bboxes.size()); ++i)
    for (int j = i + 1; j < int(bboxes.size());) {
      float minx = std::max(bboxes[i].x1, bboxes[j].x1);
      float miny = std::max(bboxes[i].y1, bboxes[j].y1);
      float maxx = std::min(bboxes[i].x2, bboxes[j].x2);
      float maxy = std::min(bboxes[i].y2, bboxes[j].y2);
      float w = std::max(float(0), maxx - minx + 1);
      float h = std::max(float(0), maxy - miny + 1);
      float interarea = w * h;
      float iou = interarea / (vArea[i] + vArea[j] - interarea);
      if (iou > nms_thresh) {
        bboxes.erase(bboxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }

    }
}

//以下为工具函数
static double sigmoid(double x) {
  return (1 / (1 + exp(-x)));
}

void YoloV5::parse_yolov5(void *output_tensor, int net_grid, float cof_threshold, std::vector<Bbox> &o_bboxes) {
  std::vector<int> anchors = get_anchors(net_grid);
  int item_size = 6;
  size_t anchor_n = 3;
  const float *output = static_cast<float *>(output_tensor);
  for (size_t n = 0; n < anchor_n; ++n)
    for (int i = 0; i < net_grid; ++i)
      for (int j = 0; j < net_grid; ++j) {
        double box_prob = output[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 4];
        box_prob = sigmoid(box_prob);
        //框置信度不满足则整体置信度不满足
        if (box_prob < cof_threshold)
          continue;

        //std::cout<<"box_prob: "<<box_prob<<std::endl;
        //注意此处输出为中心点坐标,需要转化为角点坐标
        double x = output[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 0];
        double y = output[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 1];
        double w = output[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 2];
        double h = output[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 3];
        //double cls_prob = sigmoid(output[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 5]);

        //  std::cout<<"cls_prob: "<<cls_prob<<std::endl;
        //double cof = box_prob * cls_prob;

        //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
        //  if(cof < cof_threshold)
        //    continue;
        // std::cout<<"cof: "<<cof<<std::endl;
        x = (sigmoid(x) * 2 - 0.5 + j) * double(input_dim_) / net_grid;
        y = (sigmoid(y) * 2 - 0.5 + i) * double(input_dim_) / net_grid;
        w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
        h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

        double r_x = x - w / 2;
        double r_y = y - h / 2;

        Bbox box(r_x, r_y, r_x + w, r_y + h, box_prob);

        //  std::cout<<box.x1<<" "<<box.y1<<" "<<box.x2<<" "<<box.y2<<" "<<box.conf<<std::endl;
        o_bboxes.push_back(box);

      }
}

void YoloV5::post(std::vector<Bbox> &bboxes, int w, int h) {
  doNMS(bboxes, nms_thresh_);

  double ratio = std::min(double(input_dim_) / w, double(input_dim_) / h);
  float dw = float((input_dim_ - ratio * w) / 2);
  float dh = float((input_dim_ - ratio * h) / 2);
  //std::cout<<ratio<<" "<<dw<<" "<<dh<<std::endl;

  for (size_t i = 0; i < bboxes.size(); ++i) {
    bboxes[i].x1 = std::max(float(0), float((bboxes[i].x1 - dw) / ratio));
    bboxes[i].y1 = std::max(float(0), float((bboxes[i].y1 - dh) / ratio));
    bboxes[i].x2 = std::min(float(w - 1), float((bboxes[i].x2 - dw) / ratio));
    bboxes[i].y2 = std::min(float(h - 1), float((bboxes[i].y2 - dh) / ratio));
  }

  return;

}


Vbbox YoloV5::postProcess(float *data0, float *data1, float *data2, int width, int height, float thresh) {
  std::vector<Bbox> output_bboxes;
  parse_yolov5(data0, 80, thresh, output_bboxes);
  parse_yolov5(data1, 20, thresh, output_bboxes);
  parse_yolov5(data2, 40, thresh, output_bboxes);

  post(output_bboxes, width, height);
  return output_bboxes;
}



