
#include "persondet.h"
#include "opencv2/opencv.hpp"
#include "../tensor.h"

#include "../common/alignface.h"

const int MAX_HEIGHT = 288;
const int MAX_WIDTH = 512;


using ImageWH = struct {
    int width;
    int height;
};


ImageWH calacOutputWH(ImageWH input) {

    ImageWH output{0, 0};
    if (input.width == 0 || input.height == 0) {
        return output;
    }

    float s = std::min((MAX_WIDTH + 0.1)  / input.width, (MAX_HEIGHT + 0.1)  / input.height);

    //四舍五入
    int nw = (s * input.width + 0.5);
    int nh = (s * input.height + 0.5);


    output.width = nw;
    output.height = nh;



    return output;
};


PersonDet::PersonDet() {}

PersonDet::~PersonDet() {

}

int PersonDet::Init(const std::string path) {
  //std::cout<<"11box_conf_threshold = "<< conf_thresh_<<std::endl;
 // conf_thresh_ = thresh;
  return engine_.Init(path);
}

int PersonDet::Process(cv::Mat &img, std::vector<cv::Rect> &rects) {
  if(img.empty()) {
    return -1;
  }
   ImageWH inputWH;
    inputWH.width = img.cols;
        inputWH.height = img.rows;

        ImageWH outputWH = calacOutputWH(inputWH);

        cv::resize(img, img, cv::Size(outputWH.width, outputWH.height));


        int half_offset_x = (MAX_WIDTH - outputWH.width) / 2;
        int half_offset_y = (MAX_HEIGHT - outputWH.height) / 2;


        cv::Mat stdPatch = cv::Mat::zeros(cv::Size(MAX_WIDTH, MAX_HEIGHT), CV_8UC3);


        img.copyTo(stdPatch(cv::Rect(half_offset_x, half_offset_y, outputWH.width, outputWH.height)));


  cc::Tensor<u_int8_t> tensor;

  //todo preprocess
  tensor.from_cvmat(stdPatch, true);

  std::vector<cc::Tensor<float>> outputTensors;
  engine_.forward<float>(tensor, outputTensors);

          auto hm_ptr = outputTensors[0].data();
        auto wh_ptr = outputTensors[1].data();
           auto reg_ptr = outputTensors[2].data();



        std::vector<int> inds;
        std::vector<std::pair<float, float>> centors;

        const int out_h = MAX_HEIGHT / 4;
        const int out_w = MAX_WIDTH / 4;


        //查找满足条件的中心点坐标
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w ; j++) {
                auto id = i * out_w  + j;
                float tmp_hm = *(hm_ptr + id );
                float tmp_hmax = *(hm_ptr + id  + out_h * out_w );
                float tmp = (tmp_hm == tmp_hmax) * tmp_hm;

                //std::cout<<" hm = "<< tmp_hm << " hmax = "<< tmp_hmax<<std::endl;
                if (tmp > 0.1) {

                    inds.push_back(id);
                    centors.push_back(std::make_pair(j, i));

                }
            }
        }

 


        std::vector<float> scores;
        std::vector<std::pair<float, float>> wh;
        std::vector<std::pair<float, float>> reg;

        for (auto idx : inds) {
            auto score = *(hm_ptr + idx);

            scores.push_back(score);

            auto wh1 = *(wh_ptr + idx );
            auto wh2 = *(wh_ptr + idx  + out_h * out_w);

            auto reg1 = *(reg_ptr + idx );
            auto reg2 = *(reg_ptr + idx  + out_h * out_w);


            wh.emplace_back(std::make_pair(wh1, wh2));

            reg.emplace_back(std::make_pair(reg1, reg2));


        }


        const int rect_num = inds.size();

        float x_ratio = (float) img.cols / outputWH.width;
        float y_ratio = (float) img.rows / outputWH.height;

        std::cout<<"rect_num = "<< rect_num<<std::endl;

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


            rects.emplace_back(cv::Rect(x, y, width, height));
        }


  
  //postProcess;
  return 0;

}
