#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "firedet/firedet.h"

int main(int argc, char **argv){
    std::string path = argv[1];
    FireDet fire;
    fire.Init(path);

    std::string img_path = argv[2];
    cv::Mat img = cv::imread(img_path);

    std::vector<cv::Rect>  rects;
    fire.Process(img, rects);

    

    return 0;
}
