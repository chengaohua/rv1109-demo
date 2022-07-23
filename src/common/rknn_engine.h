#ifndef SRC_ENGIN_SRC_RKNN_RKNNENGIN_H_
#define SRC_ENGIN_SRC_RKNN_RKNNENGIN_H_

#include "../tensor.h"
#include <string>
#include "rknn_api.h"

namespace cc {

class RknnModelInfo {
 public:
  RknnModelInfo() = default;

  ~RknnModelInfo() noexcept = default;

 public:
  bool isValid = false;
  uint32_t inputNum = 0;
  uint32_t outputNum = 0;
  rknn_sdk_version version;
  std::vector<rknn_tensor_attr> inputAttrs;
  std::vector<rknn_tensor_attr> outputAttrs;
  std::vector<Shape> outputShapes_;

};


class RknnEngin {
 public:
  RknnEngin();

  ~RknnEngin();

  int Init(const std::string path);

  void Release();

  template <class T>
  int forward(const Tensor<u_int8_t> &inputTensor,
              std::vector<Tensor<T>> &outputTensors);

          
 protected:
  int loadRknnModel(std::string path);

  int getModelInfo();

  void printTensorAttr(rknn_tensor_attr &attr);

 protected:
  std::pair<rknn_context, bool> rknnCtx_;
  RknnModelInfo modelInfo_;
  
};
}  // namespace arctern
#endif  // SRC_ENGIN_SRC_RKNN_RKNNENGIN_H_