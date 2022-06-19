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
} cc_fire_det_handle;


int fire_det_create(cc_fire_det_handle * handle, const char * model);

//now get top 10
int fire_det_exec(const cc_fire_det_handle * handle, cc_image * img, cc_rect rects[10], int * size);

int fire_det_destroy(cc_fire_det_handle * handle);



#endif //FIRE_DEMO_SRC_INTERFACE_INTERFACE_H_
