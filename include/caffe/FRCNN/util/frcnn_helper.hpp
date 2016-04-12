// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/04/01
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_HELPER_HPP_
#define CAFFE_FRCNN_HELPER_HPP_

#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
Point4f<Dtype> bbox_transform(const Point4f<Dtype>& ex_rois,const Point4f<Dtype>& gt_rois);

template <typename Dtype>
std::vector<Point4f<Dtype> > bbox_transform(const std::vector<Point4f<Dtype> >& ex_rois,
                                 const std::vector<Point4f<Dtype> >& gt_rois);

template <typename Dtype>
Point4f<Dtype> bbox_transform_inv(const Point4f<Dtype>& box, const Point4f<Dtype>& delta);

template <typename Dtype>
std::vector<Point4f<Dtype> > bbox_transform_inv(const Point4f<Dtype>& box,
                                      const std::vector<Point4f<Dtype> >& deltas);

template <typename Dtype>
void unmap(std::vector<int> data_in, Dtype * data_out, std::vector<int> inds,
           Dtype fill, int channel, int height, int width) {
  Dtype fill_data = fill;
  std::fill(data_out, data_out + channel * height * width, fill_data);
  for (size_t i = 0; i < inds.size(); i++) {
    int idx = inds[i];
    data_out[idx] = data_in[i];
  }
}

template <typename Dtype>
void unmap(std::vector<Dtype> data_in, Dtype * data_out, std::vector<int> inds,
           Dtype fill, int channel, int height, int width) {
  Dtype fill_data = fill;
  std::fill(data_out, data_out + channel * height * width, fill_data);
  for (size_t i = 0; i < inds.size(); i++) {
    int idx = inds[i];
    data_out[idx] = data_in[i];
  }
}

template <typename Dtype>
void unmap(std::vector<Point4f<Dtype> > data_in, Dtype * data_out, std::vector<int> inds,
           Dtype fill, int channel, int height, int width) {
  Dtype fill_data = fill;
  std::fill(data_out, data_out + channel * height * width, fill_data);
  for (size_t i = 0; i < inds.size(); i++) {
    int idx = inds[i];
    idx = 3 * (idx / (height * width)) * height * width + idx;
    for (int j = 0; j < 4; j++) {
      data_out[idx + j* height * width] = data_in[i][j];
    }
  }
}

}  // namespace frcnn

}  // namespace caffe

#endif
