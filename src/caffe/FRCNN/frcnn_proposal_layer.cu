// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/FRCNN/frcnn_proposal_layer.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  DLOG(ERROR) << "========== enter proposal layer";
  const Dtype *bottom_rpn_score = bottom[0]->cpu_data();
  const Dtype *bottom_rpn_bbox = bottom[1]->cpu_data();
  const Dtype *bottom_im_info = bottom[2]->cpu_data();

  const int num = bottom[1]->num();
  const int channes = bottom[1]->channels();
  const int height = bottom[1]->height();
  const int width = bottom[1]->width();
  CHECK(num == 1) << "only single item batches are supported";
  CHECK(channes % 4 == 0) << "rpn bbox pred channels should be divided by 4";

  const float im_height = bottom_im_info[0];
  const float im_width = bottom_im_info[1];

  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float rpn_nms_thresh;
  int rpn_min_size;
  if (this->phase_ == TRAIN) {
    rpn_pre_nms_top_n = FrcnnParam::rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::rpn_nms_thresh;
    rpn_min_size = FrcnnParam::rpn_min_size;
  } else {
    rpn_pre_nms_top_n = FrcnnParam::test_rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::test_rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::test_rpn_nms_thresh;
    rpn_min_size = FrcnnParam::test_rpn_min_size;
  }
  const int config_n_anchors = FrcnnParam::anchors.size() / 4;

  typedef pair<float, int> score_idx_pair;
  vector<score_idx_pair>  score_idx_vector;

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < config_n_anchors; k++) {
        Dtype score = bottom_rpn_score[config_n_anchors * height * width +
                                       k * height * width + j * width + i];
        const int index =
            i * height * config_n_anchors + j * config_n_anchors + k;
        score_idx_vector.push_back(score_idx_pair(score, index));
      }
    }
  }

  DLOG(ERROR) << "========== generate anchors";
  std::sort(score_idx_vector.begin(), score_idx_vector.end(), std::greater<score_idx_pair>());

  int n_anchors = score_idx_vector.size();
  n_anchors = std::min(n_anchors, rpn_pre_nms_top_n);
  std::vector<Point4f<Dtype> > anchors(n_anchors);

  //
  for (size_t index = 0; index < n_anchors; index++) {
    int pick = score_idx_vector[index].second;
    int i = pick / (height * config_n_anchors);
    int j = (pick % (height * config_n_anchors)) / config_n_anchors;
    int k = pick % config_n_anchors;

    Point4f<Dtype> anchor(
        FrcnnParam::anchors[k * 4 + 0] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
        FrcnnParam::anchors[k * 4 + 1] + j * FrcnnParam::feat_stride,  // shift_y[i][j];
        FrcnnParam::anchors[k * 4 + 2] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
        FrcnnParam::anchors[k * 4 + 3] + j * FrcnnParam::feat_stride); // shift_y[i][j];

    Point4f<Dtype> box_delta(
        bottom_rpn_bbox[(k * 4 + 0) * height * width + j * width + i],
        bottom_rpn_bbox[(k * 4 + 1) * height * width + j * width + i],
        bottom_rpn_bbox[(k * 4 + 2) * height * width + j * width + i],
        bottom_rpn_bbox[(k * 4 + 3) * height * width + j * width + i]);

    anchors[index] = bbox_transform_inv(anchor, box_delta);
  }

  // clip predicted boxes to image
  DLOG(ERROR) << "========== clip boxes";
  const Dtype bounds[4] = { im_width -1, im_height -1, im_width -1, im_height -1};
  for (int i = 0; i < n_anchors; i++) {
    for (int j = 0; j < 4; j++) {
      anchors[i].Point[j] = std::max(Dtype(0), std::min(anchors[i][j], bounds[j]));
    }
  }

  std::vector<bool> select(n_anchors, true);

  // remove predicted boxes with either height or width < threshold
  int n_select = n_anchors;
  for (int i = 0; i < n_anchors; i++) {
    float min_size = bottom_im_info[2] * rpn_min_size;
    if ((anchors[i].Point[2] - anchors[i].Point[0]) < min_size ||
        (anchors[i].Point[3] - anchors[i].Point[1]) < min_size) {
      select[i] = false;
      n_select--;
    }
  }

  DLOG(ERROR) << "==========  Convert Box to int[]";
  vector<float> sorted_boxes(n_select * 4) ;
  int select_idx = 0;
  for (int i = 0; i < n_anchors; i++) {
    if (select[i]) {
      Point4f<Dtype> &anchor = anchors[i];
      for (int j = 0; j < 4; j++) {
        sorted_boxes[select_idx * 4 + j] = anchor[j];
      }
      select_idx++;
    }
  }

  // apply nms
  DLOG(ERROR) << "========== apply nms";
  vector<int> keep_out(n_select);
  int num_out;
  _nms(&keep_out[0], &num_out, &sorted_boxes[0], n_select, 4, rpn_nms_thresh);
  DLOG(ERROR) << "rpn num after nms: " << num_out;

  num_out = std::min(num_out, rpn_post_nms_top_n);
  DLOG(ERROR) << "========== copy to top";
  top[0]->Reshape(num_out, 5, 1, 1);
  Dtype *top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < num_out; i++) {
    top_data[i * 5] = 0;
    for (int j = 1; j < 5; j++) {
      int keep_idx = keep_out[i];
      top_data[i * 5 + j] = sorted_boxes[keep_idx * 4 + j - 1];
    }
  }

  DLOG(ERROR) << "========== exit proposal layer";
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FrcnnProposalLayer);

} // namespace Frcnn

} // namespace caffe
