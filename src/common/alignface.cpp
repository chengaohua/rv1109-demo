//
// Created by gh on 2022/6/19.
//

#include "opencv2/opencv.hpp"


void ULsee_rigid_transform(const cv::Mat &A, const cv::Mat &B, cv::Mat &R,
                           cv::Mat &T, float &scale) {
  assert(A.cols == B.cols);
  assert(A.rows == B.rows);
  cv::Mat meanA, centroidA, meanB, centroidB;
  cv::reduce(A, meanA, 0, CV_REDUCE_AVG);
  cv::repeat(meanA, A.rows, 1, centroidA);
  cv::Mat AA = A - centroidA;
  cv::reduce(B, meanB, 0, CV_REDUCE_AVG);
  cv::repeat(meanB, B.rows, 1, centroidB);
  cv::Mat BB = B - centroidB;
  cv::Mat H = AA.t() * BB / A.rows;
  cv::SVD decomp = cv::SVD(H);
  cv::Mat S, U, V_t;
  S = decomp.w;
  U = decomp.u;
  V_t = decomp.vt;
  if (cv::determinant(U) * cv::determinant(V_t) < 0) {
    S.row(S.rows - 1) *= -1.;
    U.col(U.cols - 1) *= -1.;
  }
  R = U * V_t;
  float varP = 0;
  //std::cout << A << std::endl;
  for (int i = 0; i < A.size().width; i++) {
    cv::Scalar mean, var;
    cv::meanStdDev(A.col(i), mean, var);
    varP += cv::sum(var * var)[0];
  }
  scale = 1.0 / varP * cv::sum(S)[0];
  T = meanB - meanA * (scale * R);
}

cv::Mat imo_cv_transform_warpper(const std::vector<cv::Point2f> &facialPoints,
                                 const std::vector<cv::Point2f> &coordPoints) {
  int len = facialPoints.size();
  float aData[len][2], bData[len][2];
  for (int i = 0; i < len; i++) {
    aData[i][0] = facialPoints[i].x;
    aData[i][1] = facialPoints[i].y;
    bData[i][0] = coordPoints[i].x;
    bData[i][1] = coordPoints[i].y;
  }
  // imo::Mat A(facialPoints), B(coordPoints);
  cv::Mat A = cv::Mat(len, 2, CV_32FC1, &aData),
      B = cv::Mat(len, 2, CV_32FC1, &bData);

  CV_Assert(A.type() == CV_32F || A.type() == CV_64F);
  cv::Mat R, T;
  float scale;
  ULsee_rigid_transform(A, B, R, T, scale);
  R = R * scale;

  cv::Mat tt;
  cv::repeat(T, len, 1, tt);
  cv::Mat diff = (A * R + tt) - B;

  cv::hconcat(R.t(), T.t(), R);

  return R;
}

cv::Mat alignFace(std::vector<cv::Point2f> &oriPoints, std::vector<cv::Point2f> & dstPoints, cv::Mat & img, int net_w, int net_h) {
  auto trans = imo_cv_transform_warpper(oriPoints, dstPoints);

  cv::Mat alignMat ;
  cv::warpAffine(img, alignMat, trans, cv::Size(net_w, net_h));

  return alignMat;

}

cv::Mat letterbox(cv::Mat & img, int net_w, int net_h) {
  //auto &img = p->image_;
  const auto input_w = net_w;
  const auto input_h = net_h;
  const auto borderValue = 114;

  cv::Mat resized;

  if (input_w == input_h) {
    int dest_w(input_w), dest_h(input_h);
    int padbottom(0), padright(0);
    int padtop = 0;
    int padleft = 0;
    if (img.cols > img.rows) {
      dest_h = static_cast<int>(img.rows * input_h / static_cast<float>(img.cols));
      padbottom = input_h - dest_h;
      padbottom = padbottom >= 0 ? padbottom : 0;
    } else {
      dest_w = static_cast<int>(img.cols * input_w / static_cast<float>(img.rows));
      padright = input_w - dest_w;
      padright = padright >= 0 ? padright : 0;
    }

    // 居中对齐
    {
      padtop = padbottom / 2;
      padleft = padright / 2;
      padbottom -= padtop;
      padright -= padleft;
    }

    cv::resize(img, resized, cv::Size(dest_w, dest_h), 0.0f, 0.0f);
    cv::copyMakeBorder(resized,
                       resized,
                       padtop,
                       padbottom,
                       padleft,
                       padright,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(borderValue, borderValue, borderValue));
  } else {
    cv::resize(img, resized, cv::Size(input_w, input_h));
  }

  return resized;
}
