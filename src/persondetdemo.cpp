#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "persondet/persondet.h"

int main(int argc, char **argv) {
  std::string path = argv[1];

  PersonDet det;
  det.Init(path, 0.1);
  cv::Mat mat = cv::imread(argv[2]);
  std::vector<cv::Rect> rects;
  std::vector<float> scores;
  std::vector<int>  cls;
  det.Process(mat, rects, scores, cls);


  return 0;
}
