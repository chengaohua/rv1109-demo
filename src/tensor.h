/****************************************************************************
 *  Copyright:      Copyright © 2020 intellifusion Inc. All rights Reserved
 *  Description:
 *  author:         chengaohua
 *  Last modified:  2020.08.25
 *  email:          chen.gaohua@intellif.com
 ****************************************************************************/

#ifndef SRC_COMMON_TENSOR_H_
#define SRC_COMMON_TENSOR_H_
#include "tensorShape.h"
#include <memory>
#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"

namespace cc {

template <typename T>
class Tensor {
 public:
  Tensor() {}

  explicit Tensor(const Shape& shape) : shape_(shape) {
    if (size() > 0) {
      data_.reset(new T[size()]());
    }
  }

  virtual ~Tensor() {}

  uint size() const { return shape_.size(); }

  const Shape& shape() { return shape_; }

  Tensor& reshape(const Shape& shape) {
    uint oldSize = shape_.size();
    uint newSize = shape.size();

    shape_ = shape;

    if (oldSize != newSize) {
      data_.reset(new T[size()]());
    }

    return *this;
  }

  const T* data() const { return &(data_.get()[0]); }

  T* data() { return &(data_.get()[0]); }

  const Shape& shape() const { return shape_; }

  void from_cvmat(const cv::Mat& mat, bool bswap_channels = true) {
    if (bswap_channels) {
      shape_ = Shape(1, mat.channels(), mat.rows, mat.cols);
      data_.reset(new T[size()]());
      data_format_ = DataFormat::NCHW;
      swap_channels(mat);
    } else {
      shape_ = Shape(1, mat.rows, mat.cols, mat.channels());
      data_.reset(new T[size()]());
      data_format_ = DataFormat::NHWC;
      memcpy((uchar*)data_.get(), (uchar*)mat.data, size() * sizeof(T));
    }
  }

  void from_cvmat(const std::vector<cv::Mat>& mats,
          int batchNum, bool bswap_channels = true)
  {
    assert((batchNum > 0) && (mats.size() <= batchNum));

    const int n = batchNum;
    const int c = mats[0].channels();
    const int h = mats[0].rows;
    const int w = mats[0].cols;
    const int nitems = c * h * w;

    if (bswap_channels) {
      shape_ = Shape(n, c, h, w);
      data_.reset(new T[size()]());
      data_format_ = DataFormat::NCHW;

      int offset = 0;
      for (size_t i = 0; i < mats.size(); ++i) {
        swap_channels(mats[i], offset);
        offset += nitems;
      }
    } else {
      shape_ = Shape(n, h, w, c);
      data_.reset(new T[size()]());
      data_format_ = DataFormat::NHWC;

      T* pdata = data_.get();
      for (size_t i = 0; i < mats.size(); ++i) {
        memcpy((uchar*)pdata, (uchar*)mats[i].data, nitems * sizeof(T));
        pdata += nitems;
      }
    }

    assert(batchNum * c * h * w == (int)this->size());
  }

  void from_vector(const std::vector<T>& vec) {
    data_.reset(new T[vec.size()]());
    memcpy((uchar*)data_.get(), (uchar*)vec.data(), vec.size() * sizeof(T));
  }

 protected:
  void swap_channels(const cv::Mat& image) {
    int width = image.cols;
    int height = image.rows;
    int depth = image.depth();



    T* pdata = data_.get();
    std::vector<cv::Mat> input_channels;

    for (int i = 0; i < image.channels(); ++i) {
      cv::Mat imgchannel(height, width, depth, pdata);
      input_channels.push_back(imgchannel);
      pdata += (height * width);
    }
    cv::split(image, input_channels);
  }


  void swap_channels(const cv::Mat& image, int offset) {
    int width = image.cols;
    int height = image.rows;
    int depth = image.depth();



    T* pdata = data_.get() + offset;
    std::vector<cv::Mat> input_channels;

    for (int i = 0; i < image.channels(); ++i) {
      cv::Mat imgchannel(height, width, depth, pdata);
      input_channels.push_back(imgchannel);
      pdata += (height * width);
    }

    cv::split(image, input_channels);
  }

  Shape shape_;
  std::shared_ptr<T> data_;
  DataFormat data_format_ = DataFormat::NCHW;
};



}  // namespace cc
#endif  // SRC_COMMON_TENSOR_H_
