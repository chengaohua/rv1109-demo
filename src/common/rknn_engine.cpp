
#include "rknn_engine.h"
#include <fstream>


using cc::RknnEngin;


RknnEngin::RknnEngin() : rknnCtx_(0, false) {
  return;
}

RknnEngin::~RknnEngin() {
  Release();
}

void RknnEngin::Release() {
  if (rknnCtx_.second) {
    rknn_destroy(rknnCtx_.first);
    rknnCtx_.second = false;
  }
}

int RknnEngin::loadRknnModel(const std::string path) {

    std::ifstream t(path);  
    std::stringstream buffer;  
    buffer << t.rdbuf();  
    std::string contents(buffer.str());


  rknn_context ctx;
  int ret = rknn_init(&ctx, (void *)contents.data(), contents.size(), 0);
  rknnCtx_.first = ctx;
  if (ret < 0) {
    rknnCtx_.second = false;
  } else {
    rknnCtx_.second = true;
  }

  return ret;
}

int RknnEngin::getModelInfo() {

  rknn_context ctx = rknnCtx_.first;
  // Query rknn sdk version
  int ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION,
                       &modelInfo_.version, sizeof(modelInfo_.version));
  if (ret != RKNN_SUCC) {
    return -1;
  }

  // Get Model Input Output Info
  rknn_input_output_num io_num = {0};
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    //SPDLOG_ERROR("rknn_query fail, ret = {}!!!", ret);
    return -1;
  }

  if ((io_num.n_input < 1) || (io_num.n_output < 1)) {
    //SPDLOG_ERROR("invalid rknn model!!!");
    return -1;
  }

  modelInfo_.inputNum = io_num.n_input;
  modelInfo_.outputNum = io_num.n_output;

  modelInfo_.inputAttrs.resize(io_num.n_input);
  modelInfo_.outputAttrs.resize(io_num.n_output);

  for (uint32_t i = 0; i < io_num.n_input; i++) {
    modelInfo_.inputAttrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR,
                     &(modelInfo_.inputAttrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
     // SPDLOG_ERROR("rknn_query fail, ret = {}!!!", ret);
      return -1;
    }
    printTensorAttr(modelInfo_.inputAttrs[i]);
  }

  for (uint32_t i = 0; i < io_num.n_output; i++) {
    modelInfo_.outputAttrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR,
                     &(modelInfo_.outputAttrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
     // SPDLOG_ERROR("rknn_query fail, ret = {}!!!", ret);
      return -1;
    }
    printTensorAttr(modelInfo_.outputAttrs[i]);
  }

  return 0;
}

void RknnEngin::printTensorAttr(rknn_tensor_attr &attr) {
  std::cout << "index=" << attr.index << " name=" << attr.name
            << " n_dims=" << attr.n_dims << " dims=[" << attr.dims[3] << " "
            << attr.dims[2] << " " << attr.dims[1] << " " << attr.dims[0] << "]"
            << " n_elems=" << attr.n_elems << " size=" << attr.size << " fmt="
            << attr.fmt << " type=" << attr.type << " qnt_type=" << attr.qnt_type
            << " fl=" << attr.fl << " zp=" << attr.zp << " scale=" << attr.scale
            << std::endl;
}

using namespace cc;
static cc::TensorShape convertRknnTensorAttr2TensorShape(rknn_tensor_attr & attr) {
  if(attr.n_dims == 4) {
    TensorShape shape(attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3]);
    return shape;
  }else if (attr.n_dims == 3) {
    TensorShape shape(1, attr.dims[0], attr.dims[1], attr.dims[2]);
    return shape;
  } else if(attr.n_dims == 2) {
    TensorShape shape(1,1, attr.dims[0], attr.dims[1]);
    return shape;
  } else {
    TensorShape shape(1,1,1,attr.dims[0]);
    return shape;
  }
}

int RknnEngin::Init(const std::string path) {

  int code = loadRknnModel(path);
  if (0 != code) {
    return -1;
  }

  int ret = getModelInfo();
  if (ret != 0) {
    return -1;
  }

  for(auto &tensorAttr : modelInfo_.outputAttrs) {
    modelInfo_.outputShapes_.emplace_back(convertRknnTensorAttr2TensorShape(tensorAttr));
  }

  return 0;
}

template <class T>
int RknnEngin::forward(const Tensor<u_int8_t> &inputTensor,
                       std::vector<Tensor<T>> &outputTensors) {

  rknn_context  ctx = rknnCtx_.first;

  // Set Input Data
  rknn_input inputs[1];

  inputs[0].index = modelInfo_.inputAttrs[0].index;
  inputs[0].pass_through = modelInfo_.inputAttrs[0].type == RKNN_TENSOR_UINT8;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = inputTensor.size();
  inputs[0].fmt = RKNN_TENSOR_NCHW;
  inputs[0].buf = (void *) inputTensor.data();

  int ret = rknn_inputs_set(ctx, 1, inputs);
  if (ret < 0) {
    //SPDLOG_ERROR("rknn_input_set fail, ret={}!!!", ret);
    return -1;
  }

  // Run
  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    //SPDLOG_ERROR("rknn_run fail, ret={}!!!", ret);
    return -1;
  }

  for(uint32_t i = 0 ; i < modelInfo_.outputNum; i++) {
    outputTensors.emplace_back(Tensor<T>(modelInfo_.outputShapes_[i]));
  }

  rknn_output outputs[modelInfo_.outputNum];
  for(uint32_t i = 0 ; i < modelInfo_.outputNum; i++) {
    outputs[i].want_float = sizeof(T) == 4;
    outputs[i].is_prealloc = 1;
    outputs[i].index = modelInfo_.outputAttrs[i].index;
    outputs[i].buf = outputTensors[i].data();
    outputs[i].size = outputTensors[i].size() * sizeof(T);
  }

  // Get Output
  ret = rknn_outputs_get(ctx, modelInfo_.outputNum, outputs, NULL);
  if (ret < 0) {
    //SPDLOG_ERROR("rknn_outputs_get fail, ret={}!!!", ret);
    return -1;
  }

  // Release rknn_outputs
  rknn_outputs_release(ctx, modelInfo_.outputNum, outputs);

  return 0;
}


template int RknnEngin::forward(const Tensor<u_int8_t> &inputTensor,
                       std::vector<Tensor<uint8_t>> &outputTensors);

template int RknnEngin::forward(const Tensor<u_int8_t> &inputTensor,
                       std::vector<Tensor<float>> &outputTensors);