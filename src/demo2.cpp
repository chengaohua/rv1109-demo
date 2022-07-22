#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "persondet/persondet.h"

int main(int argc, char **argv) {
  std::string path = argv[1];

  PersonDet det;
  det.Init(path);
  cv::Mat mat = cv::imread(argv[2]);
  std::vector<cv::Rect> rects;
  det.Process(mat, rects);



  return 0;
}
