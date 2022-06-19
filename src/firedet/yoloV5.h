
#ifndef ARCTERN_BASE_SRC_BASE_YOLOV5_H_
#define ARCTERN_BASE_SRC_BASE_YOLOV5_H_

#include <vector>

struct Bbox {
  Bbox(int xx1, int yy1, int xx2, int yy2, float box_score, float dxx1 = 0.,
       float dyy1 = 0., float dxx2 = 0., float dyy2 = 0.)
      : x1(xx1),
        y1(yy1),
        x2(xx2),
        y2(yy2),
        score(box_score),
        dx1(dxx1),
        dy1(dyy1),
        dx2(dxx2),
        dy2(dyy2) {}

  float x1, y1, x2, y2;
  float score;
  float dx1, dy1, dx2, dy2;
  int label;

  float quality;
};

typedef std::vector<Bbox> Vbbox;


class YoloV5 final  {
 public:
  YoloV5() = default;
  ~YoloV5() = default;

  std::vector<int> get_anchors(int net_grid);

  Vbbox postProcess(float * data0,float * data1, float * data2, int width, int height, float thresh);
  
  void post(std::vector<Bbox>& bboxes, int w, int h);

  //注意此处的阈值是框和物体prob乘积的阈值
  void parse_yolov5(void* output_tensor,int net_grid, float cof_threshold, std::vector<Bbox>& o_bboxes);
  void doNMS(std::vector<Bbox>& bboxes, float nms_thresh);
  void setNmsThresh(float thresh){
  	nms_thresh_ = thresh;
  }
 private:
  int input_dim_ = 640;
  int input_dim2_ = 640;
  float nms_thresh_ = 0.45;
};


#endif //ARCTERN_BASE_SRC_BASE_YOLOV5_H_
