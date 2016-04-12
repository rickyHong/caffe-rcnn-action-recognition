#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

INSTANTIATE_CLASS(Point4f);
INSTANTIATE_CLASS(BBox);

template <typename Dtype>
Dtype get_iou(const Point4f<Dtype> &A, const Point4f<Dtype> &B) {
  const Dtype xx1 = std::max(A[0], B[0]);
  const Dtype yy1 = std::max(A[1], B[1]);
  const Dtype xx2 = std::min(A[2], B[2]);
  const Dtype yy2 = std::min(A[3], B[3]);
  Dtype inter = std::max(Dtype(0), xx2 - xx1 + 1) * std::max(Dtype(0), yy2 - yy1 + 1);
  Dtype areaA = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
  Dtype areaB = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
  return inter / (areaA + areaB - inter);
}
template float get_iou(const Point4f<float> &A, const Point4f<float> &B);
template double get_iou(const Point4f<double> &A, const Point4f<double> &B);

template <typename Dtype>
vector<Dtype> get_ious(const vector<Point4f<Dtype> > &A, const vector<Point4f<Dtype> > &B) {
  vector<Dtype> ious;
  for (size_t i = 0; i < A.size(); i++) {
    for (size_t j = 0; j < B.size(); j++) {
      ious.push_back(get_iou(A[i], B[j]));
    }
  }
  return ious;
}
template vector<float> get_ious(const vector<Point4f<float> > &A, const vector<Point4f<float> > &B);
template vector<double> get_ious(const vector<Point4f<double> > &A, const vector<Point4f<double> > &B);

template <typename Dtype>
void draw_bbox(cv::Mat &frame, const std::vector<BBox<Dtype> > &bboxs) {
  for (size_t i = 0; i < bboxs.size(); i++) {
    cv::rectangle(frame, cv::Point(bboxs[i][0], bboxs[i][1]),
                  cv::Point(bboxs[i][2], bboxs[i][3]), cv::Scalar(255, 0, 0));
    const std::string contxt = bboxs[i].to_short_string();
    cv::putText(frame, contxt.c_str(), cv::Point(bboxs[i][0], bboxs[i][1]), 0,
                0.4, cv::Scalar(0, 255, 0));
  }
}
template void draw_bbox(cv::Mat &frame, const std::vector<BBox<float> > &bboxs); 
template void draw_bbox(cv::Mat &frame, const std::vector<BBox<double> > &bboxs); 

float get_scale_factor(int width, int height, int short_size, int max_long_size) {
  float im_size_min = std::min(width, height);
  float im_size_max = std::max(width, height);

  float scale_factor = static_cast<float>(short_size) / im_size_min;
  // Prevent the biggest axis from being more than max_size
  if (scale_factor * im_size_max > max_long_size) {
    scale_factor = static_cast<float>(max_long_size) / im_size_max;
  }
  return scale_factor;
}

template <typename Dtype>
void get_max_idx(const Dtype * const data, int n_col, Dtype& max_val, int& max_idx, int stride = 1) {
  max_idx = 0;
  max_val = data[0];
  for (int i = 0; i < n_col; i++) {
    if (data[i*stride] > max_val) {
      max_val = data[i*stride];
      max_idx = i;
    }
  }
}

template <typename Dtype>
void get_max_idxs(const vector<Dtype>& data, int n_col, vector<Dtype>& max_vals, vector<int>& max_idxs, int axis) {
  if (data.size() == 0) return;
  if (n_col <= 0) {
    n_col = data.size();
  }
  CHECK(data.size() % n_col == 0) << "wrong column numer";

  if (axis == 0) {
    for (size_t i = 0; i * n_col < data.size(); i++) {
      int max_idx;
      Dtype max_val;
      get_max_idx(&data[i * n_col], n_col, max_val, max_idx);
      max_vals.push_back(max_val);
      max_idxs.push_back(max_idx);
    }
  } else {
    int n_row = data.size()/n_col;
    for (int i = 0; i < n_col ; i++) {
      int max_idx;
      Dtype max_val;
      get_max_idx(&data[i], n_row, max_val, max_idx, n_col);
      max_vals.push_back(max_val);
      max_idxs.push_back(max_idx);
    }
  }
}
template void get_max_idxs(const vector<float>& data, int n_col, vector<float>& max_vals, vector<int>& max_idxs, int axis);
template void get_max_idxs(const vector<double>& data, int n_col, vector<double>& max_vals, vector<int>& max_idxs, int axis);

template <typename Dtype>
std::vector<int> get_equal_idx(const std::vector<Dtype> data_vector, Dtype target, int start_idx, int stride) {
  std::vector<int> idx_vector;
  const int vec_size = data_vector.size(); 
  for (int i = 0; i * stride + start_idx < vec_size; i = i + 1) {
    int idx = start_idx + i * stride;
    if (data_vector[idx] == target) {
      idx_vector.push_back(i);
    }   
  }
  return idx_vector;
}
template std::vector<int> get_equal_idx(const std::vector<float> data_vector, float target, int start_idx = 0, int stride = 1);
template std::vector<int> get_equal_idx(const std::vector<double> data_vector, double target, int start_idx = 0, int stride = 1);
template std::vector<int> get_equal_idx(const std::vector<int> data_vector, int target, int start_idx = 0, int stride = 1);

} // namespace frcnn

} // namespace caffe
