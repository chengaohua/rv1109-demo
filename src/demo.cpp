#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "interface/interface.h"

int main(int argc, char **argv) {
  std::string path = argv[1];

  cc_fire_det_handle det_handle;
  //可信度阈值设置
  float thresh = 0.1;
  fire_det_create(&det_handle, argv[1] , thresh);

  //det loop
  do {
    cv::Mat img = cv::imread(argv[2]);
    cc_image image;
    image.data = (const char *) img.data;
    image.width = img.cols;
    image.height = img.rows;
    image.format = CC_IMAGE_BGR888;

    cc_rect rects[10];
    int size = 0;
    fire_det_exec(&det_handle, &image, rects, &size);

    for(int i = 0 ; i < size; i++) {
      printf("[%f,%f,%f,%f] cls = %d conf = %f\n", rects[i].x, rects[i].y, rects[i].width, rects[i].height, rects[i].cls, rects[i].conf);
    }

    

    printf("det fire nums = %d\n", size);

    break;
  } while (-1);

  fire_det_destroy(&det_handle);

  return 0;
}
