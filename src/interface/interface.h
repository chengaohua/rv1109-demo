//
// Created by gh on 2022/6/19.
//

#ifndef FIRE_DEMO_SRC_INTERFACE_INTERFACE_H_
#define FIRE_DEMO_SRC_INTERFACE_INTERFACE_H_

/**
 * 图片数据格式枚举
 */
typedef enum {
  CC_IMAGE_BGR888 = 0,
  CC_IMAGE_NV21,
  CC_IMAGE_NV12,
}cc_image_format;


/**
 * cc点数据结构
 */
typedef struct cc_point {
  float x;
  float y;
} cc_point;


/**
 * cc框数据结构
 */
typedef struct cc_rect {
  float x;
  float y;
  float width;
  float height;
  float conf;  // 框可信度
  int  cls;    // 类别
} cc_rect;

/**
 * imo图片数据结构
 */
typedef struct cc_image {
  /**
   * 图片数据
   */
  const char *data;
  /**
   * 图片的宽
   */
  int width;
  /**
   * 图片的高
   */
  int height;
  /**
   * 图片数据的颜色空间
   */
  cc_image_format format;
} cc_image;

typedef struct {
  /*
   *
   * */
  void * handle;
} cc_det_handle;




int fire_det_create(cc_det_handle * handle, const char * model,float conf_thresh);

//now get top 10
//@return  cls=0 is fire;  cls = 1 is smoke. 
int fire_det_exec(const cc_det_handle * handle, cc_image * img, cc_rect rects[10], int * size);

int fire_det_destroy(cc_det_handle * handle);

int stationery_det_create(cc_det_handle * handle, const char * model,float conf_thresh, float nms_thresh);

int stationery_det_exec(const cc_det_handle * handle, cc_image * img, cc_rect rects[100], int * size);

int stationery_det_destroy(cc_det_handle * handle);

int multi_det_create(cc_det_handle * handle, const char * model,float conf_thresh);


/*
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
*/
int multi_det_exec(const cc_det_handle * handle, cc_image * img, cc_rect rects[100], int * size);

int multi_det_destroy(cc_det_handle * handle);



#endif //FIRE_DEMO_SRC_INTERFACE_INTERFACE_H_
