

#include "Yolo.h"

#include <iostream>
#include <sys/time.h>


std::vector<BoxInfo>
decode_infer(cc::Tensor<float> & tensor, int stride, const yolocv::YoloSize &frame_size, int net_size, int num_classes,
                     const std::vector<yolocv::YoloSize> &anchors, float threshold) 
{
    std::vector<BoxInfo> result;
    int batchs, channels, height, width, pred_item ;
    batchs = tensor.shape()[4];
    channels = tensor.shape()[3];
    height = tensor.shape()[2];
    width = tensor.shape()[1];
    pred_item = tensor.shape()[0];

    printf("batchs = %d\n", batchs);
    printf("channels = %d\n", channels);
    printf("height = %d\n", height);
    printf("width = %d\n", width);
    printf("pred_item = %d\n", pred_item);

   

    // std::vector<float> tmp_buffer(batchs * channels * height * width * pred_item);
    // for(int ci = 0 ; ci < channels; ci++) {
    //     for(int hi = 0; hi < height; hi++) {
    //         for(int wi = 0 ; wi < width; wi++) {
    //             for(int pi = 0; pi < pred_item; pred_item++) {
    //                 tmp_buffer[ci * height * width*pred_item + hi * width * pred_item + wi * pred_item + pi] = 
    //                 tensor.data()[pi * width * height * channels   + wi * height * channels + hi * channels + channels ]; 
    //             }
    //         }
    //     }
    // }

    //  float * data_ptr = tmp_buffer.data();

   float * data_ptr = tensor.data();
    for(int bi=0; bi<batchs; bi++)
    {
        //auto batch_ptr = data_ptr + bi*(channels*height*width*pred_item);
        for(int ci=0; ci<channels; ci++)
        {
            //auto channel_ptr = batch_ptr + ci*(height*width*pred_item);
            for(int hi=0; hi<height; hi++)
            {
              //  auto height_ptr = channel_ptr + hi*(width * pred_item);
                for(int wi=0; wi<width; wi++)
                {
                //    auto width_ptr = height_ptr + wi*pred_item;
                   // auto cls_ptr = width_ptr + 5;
                    auto conf_tmp = data_ptr[4 * width * height * channels + wi * height * channels + hi * channels + channels];

                    auto confidence = sigmoid(conf_tmp);

                    std::cout<<"confidence = "<< confidence<<std::endl;

                    for(int cls_id=0; cls_id<num_classes; cls_id++)
                    {
                        float tmp = sigmoid(data_ptr[(5+ cls_id) * width *height * channels + wi * height * channels + hi * channels + channels]);
                        float score = tmp * confidence;
                         std::cout<<tmp<<" score = "<< score<<std::endl;
                        if(score > threshold)
                        {
                            float cx = (sigmoid(data_ptr[0+ wi * height * channels + hi * channels + channels]) * 2.f - 0.5f + wi) * (float) stride;
                            float cy = (sigmoid(data_ptr[1 * width * height * channels + wi * height * channels + hi * channels + channels]) * 2.f - 0.5f + hi) * (float) stride;
                            float w = pow(sigmoid(data_ptr[2 * width * height * channels + wi * height * channels + hi * channels + channels]) * 2.f, 2) * anchors[ci].width;
                            float h = pow(sigmoid(data_ptr[3 * width * height * channels + wi * height * channels + hi * channels + channels]) * 2.f, 2) * anchors[ci].height;
                            
                            BoxInfo box;
                            
                            box.x1 = std::max(0, std::min(frame_size.width, int((cx - w / 2.f) )));
                            box.y1 = std::max(0, std::min(frame_size.height, int((cy - h / 2.f) )));
                            box.x2 = std::max(0, std::min(frame_size.width, int((cx + w / 2.f) )));
                            box.y2 = std::max(0, std::min(frame_size.height, int((cy + h / 2.f) )));
                            box.score = score;
                            box.label = cls_id;
                            result.push_back(box);
                        }
                    }
                }
            }
        }
    }

    return result;
}

void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}


// std::vector<BoxInfo>
// decode_infer(cc::Tensor<float> & data, int stride, const yolocv::YoloSize &frame_size, int net_size, int num_classes,
//                      const std::vector<yolocv::YoloSize> &anchors, float threshold) 
// {
//     std::vector<BoxInfo> result;
//     int batchs, channels, height, width, pred_item ;
//     batchs = data.shape()[0];
//     channels = data.shape()[1];
//     height = data.shape()[2];
//     width = data.shape()[3];
//     pred_item = data.shape()[4];

//     auto data_ptr = data.host<float>();
//     for(int bi=0; bi<batchs; bi++)
//     {
//         auto batch_ptr = data_ptr + bi*(channels*height*width*pred_item);
//         for(int ci=0; ci<channels; ci++)
//         {
//             auto channel_ptr = batch_ptr + ci*(height*width*pred_item);
//             for(int hi=0; hi<height; hi++)
//             {
//                 auto height_ptr = channel_ptr + hi*(width * pred_item);
//                 for(int wi=0; wi<width; wi++)
//                 {
//                     auto width_ptr = height_ptr + wi*pred_item;
//                     auto cls_ptr = width_ptr + 5;

//                     auto confidence = sigmoid(width_ptr[4]);

//                     for(int cls_id=0; cls_id<num_classes; cls_id++)
//                     {
//                         float score = sigmoid(cls_ptr[cls_id]) * confidence;
//                         if(score > threshold)
//                         {
//                             float cx = (sigmoid(width_ptr[0]) * 2.f - 0.5f + wi) * (float) stride;
//                             float cy = (sigmoid(width_ptr[1]) * 2.f - 0.5f + hi) * (float) stride;
//                             float w = pow(sigmoid(width_ptr[2]) * 2.f, 2) * anchors[ci].width;
//                             float h = pow(sigmoid(width_ptr[3]) * 2.f, 2) * anchors[ci].height;
                            
//                             BoxInfo box;
                            
//                             box.x1 = std::max(0, std::min(frame_size.width, int((cx - w / 2.f) )));
//                             box.y1 = std::max(0, std::min(frame_size.height, int((cy - h / 2.f) )));
//                             box.x2 = std::max(0, std::min(frame_size.width, int((cx + w / 2.f) )));
//                             box.y2 = std::max(0, std::min(frame_size.height, int((cy + h / 2.f) )));
//                             box.score = score;
//                             box.label = cls_id;
//                             result.push_back(box);
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     return result;
// }