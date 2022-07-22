//
// Created by DefTruth on 2021/11/6.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_YOLOV5_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_YOLOV5_H

#include <memory>
#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>

namespace types {
  struct Boxf {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
    int id;
    const char * label_text;
    bool flag;
  };
}

namespace mnncv
{
  class  MNNYoloV5 
  {
  public:
    explicit MNNYoloV5(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNYoloV5()  = default;

  private:
    // nested classes
    typedef struct
    {
      float w_r;
      float h_r;
      int dw;
      int dh;
      int new_unpad_w;
      int new_unpad_h;
      bool flag;
    } YoloV5ScaleParams;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.0 / 255.f, 1.0 / 255.f, 1.0 / 255.f};
    const char *class_names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

    int input_width = 640;
    int input_height = 640;

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat_rs) ; // without resize

    std::shared_ptr<MNN::CV::ImageProcess> pretreat; // init at subclass
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::Tensor *input_tensor = nullptr; // assume single input.
    //MNN::ScheduleConfig schedule_config;

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YoloV5ScaleParams &scale_params);

    void generate_bboxes(const YoloV5ScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         const std::map<std::string, MNN::Tensor *> &output_tensors,
                         float score_threshold, int img_height,
                         int img_width); // rescale & exclude

    // void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
    //          float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.5f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_YOLOV5_H
