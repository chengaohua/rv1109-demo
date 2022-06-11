
#ifndef SRC_COMMON_TENSORSHAPE_H_
#define SRC_COMMON_TENSORSHAPE_H_

#include <iostream>
#include <vector>
namespace cc {
enum DataFormat { NHWC = 0, NCHW, NTHWC, NCTHW };

template<int N>
class Shape {
 public:
  Shape() {
    for (int i = 0; i < N; ++i) {
      val[i] = static_cast<int>(0);
    }

    kdims = N;
  }

  explicit Shape(int v0) : kdims(1) { val[0] = v0; }

  Shape(int v0, int v1) : kdims(2) {
    val[0] = v0;
    val[1] = v1;
  }

  Shape(int v0, int v1, int v2) : kdims(3) {
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
  }

  Shape(int v0, int v1, int v2, int v3) : kdims(4) {
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
  }

  Shape(int v0, int v1, int v2, int v3, int v4) : kdims(5) {
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
  }

  explicit Shape(std::vector<int> &shapes) {
    kdims = shapes.size();
    if (shapes.size() > 0) val[0] = shapes[0];
    if (shapes.size() > 1) val[1] = shapes[1];
    if (shapes.size() > 2) val[2] = shapes[2];
    if (shapes.size() > 3) val[3] = shapes[3];
    if (shapes.size() > 4) val[4] = shapes[4];
  }

  int &operator[](int i) { return val[i]; }

  int operator[](int i) const { return val[i]; }

  int num_dims() const { return kdims; }

  bool operator==(const Shape &rhs) {
    if (num_dims() != rhs.num_dims()) return false;

    int i = 0;
    for (; i < num_dims() && (val[i] == rhs.val[i]); ++i) {
    }

    return (i == num_dims());
  }

  uint size() const {
    if (val[0] <= 0) { return 0; }

    uint s = 1;
    for (int i = 0; i < num_dims(); ++i) {
      if (val[i] > 0) {
        s *= val[i];
      } else {
        break;
      }
    }
    return s;
  }

  template<int M>
  friend inline std::ostream &operator<<(std::ostream &os, const Shape<M> &shape);

 protected:
  int val[N];

 private:
  int kdims = 0;  // available dimensions
};

template<int N>
inline std::ostream &operator<<(std::ostream &os, const Shape<N> &shape) {
//  const char * beg =  "shape dims:  ( ";
//  os<<beg;
  for (int i = 0; i < shape.num_dims(); ++i) {
    os << shape[i] << " ";
  }
//  const char * end = ")";
//  os << end;
  return os;
}

typedef Shape<4> TensorShape;
}

#endif //SRC_COMMON_TENSORSHAPE_H_
