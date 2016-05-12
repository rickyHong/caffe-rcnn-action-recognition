// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <iomanip>
#include "caffe/FRCNN/frcnn_anchor_target_layer.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnAnchorTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                               const vector<Blob<Dtype> *> &top) {
  anchors_ = FrcnnParam::anchors;
  config_n_anchors_ = FrcnnParam::anchors.size() / 4;
  feat_stride_ = FrcnnParam::feat_stride;
  border_ = FrcnnParam::rpn_allowed_border;

  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // labels
  top[0]->Reshape(1, 1, config_n_anchors_ * height, width);
  // bbox_targets
  top[1]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_inside_weights
  top[2]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_outside_weights
  top[3]->Reshape(1, config_n_anchors_ * 4, height, width);

  LOG(INFO) << "FrcnnAnchorTargetLayer : " << config_n_anchors_ << " anchors , " << feat_stride_ << " feat_stride , " << border_ << " allowed_border";
  LOG(INFO) << "FrcnnAnchorTargetLayer : " << this->layer_param_.name() << " SetUp";
}

template <typename Dtype>
void FrcnnAnchorTargetLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  DLOG(ERROR) << "========== enter anchor target layer";

  const Dtype *bottom_gt_bbox = bottom[1]->cpu_data();
  const Dtype *bottom_im_info = bottom[2]->cpu_data();

  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  CHECK(num == 1) << "only single item batches are supported";

  const Dtype im_height = bottom_im_info[0];
  const Dtype im_width = bottom_im_info[1];

  // gt boxes (x1, y1, x2, y2, label)
  vector<Point4f<Dtype> > gt_boxes;
  for (int i = 0; i < bottom[1]->num(); i++) {
    const Dtype * base_address = &bottom_gt_bbox[(i * bottom[1]->channels())];
    gt_boxes.push_back(Point4f<Dtype>(base_address[0], base_address[1], base_address[2],
                            base_address[3]));
  }

  // Generate anchors
  DLOG(ERROR) << "========== generate anchors";
  vector<int> inds_inside;
  vector<Point4f<Dtype> > anchors;

  Dtype bounds[4] = {-border_, -border_, im_width + border_, im_height + border_};

  for (int k = 0; k < config_n_anchors_; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        float x1 = j * feat_stride_ + anchors_[k * 4 + 0];  // shift_x[i][j];
        float y1 = i * feat_stride_ + anchors_[k * 4 + 1];  // shift_y[i][j];
        float x2 = j * feat_stride_ + anchors_[k * 4 + 2];  // shift_x[i][j];
        float y2 = i * feat_stride_ + anchors_[k * 4 + 3];  // shift_y[i][j];
        if (x1 >= bounds[0] && y1 >= bounds[1] && x2 < bounds[2] &&
            y2 < bounds[3]) {
          inds_inside.push_back(k * height  * width + i * width + j);
          anchors.push_back(Point4f<Dtype>(x1, y1, x2, y2));
        }
      }
    }
  }

  const int n_anchors = anchors.size();

  // label: 1 is positive, 0 is negative, -1 is dont care
  vector<int> labels(n_anchors, -1);

  vector<Dtype> max_overlaps;
  vector<int> argmax_overlaps;
  vector<Dtype> gt_max_overlaps;
  vector<int> gt_argmax_overlaps;

  if (gt_boxes.size() > 0) {
    vector<Dtype> ious = get_ious(anchors, gt_boxes);
    get_max_idxs(ious, gt_boxes.size(), max_overlaps, argmax_overlaps);
    get_max_idxs(ious, gt_boxes.size(), gt_max_overlaps, gt_argmax_overlaps, 1);
    std::set<int> gt_argmax_set;
    for (size_t i = 0; i < gt_max_overlaps.size(); i ++) {
      if (gt_max_overlaps[i] < FrcnnParam::rpn_positive_overlap) {
        DLOG(ERROR) << gt_max_overlaps[i] << ":gt--" << gt_boxes[i].to_string()
                    << "  anchor--" << anchors[gt_argmax_overlaps[i]].to_string();
      }

      vector<int> tmp_idxs = get_equal_idx(ious, gt_max_overlaps[i], i, gt_max_overlaps.size());
      for (int j = 0; j < tmp_idxs.size(); j ++) {
        gt_argmax_set.insert(tmp_idxs[j]);
      }
    }
    gt_argmax_overlaps = std::vector<int>(gt_argmax_set.begin(), gt_argmax_set.end());
  } else {
    max_overlaps = vector<Dtype>(n_anchors, Dtype(0));
  }

  for (int i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] < FrcnnParam::rpn_negative_overlap) {
      labels[i] = 0;
    }
    if (max_overlaps[i] >= FrcnnParam::rpn_positive_overlap) {
      labels[i] = 1;
    }
  }

  DLOG(ERROR) << "label == 1: " << get_equal_idx(labels, 1).size();
  DLOG(ERROR) << "label == 0: " << get_equal_idx(labels, 0).size();
  DLOG(ERROR) << "gt_argmax_overlaps: " << gt_argmax_overlaps.size();

  DLOG(ERROR) << "========== assign labels";
  // fg label: for each gt, anchor with highest overlap
  for (int i = 0; i < gt_argmax_overlaps.size() ; ++i) {
    labels[gt_argmax_overlaps[i]] = 1;
  }

  // subsample positive labels if we have too many
  int num_fg = FrcnnParam::rpn_fg_fraction * FrcnnParam::rpn_batchsize;
  vector<int> fg_inds = get_equal_idx(labels, 1);

  DLOG(ERROR) << "========== supress_positive labels";
  if (fg_inds.size() > num_fg) {
    std::set<int> ind_set;
    while (ind_set.size() < fg_inds.size() - num_fg) {
      int tmp_idx = caffe::caffe_rng_rand() % fg_inds.size();
      ind_set.insert(fg_inds[tmp_idx]);
    }
    for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end(); it++) {
      labels[*it] = -1;
    }
  }

  DLOG(ERROR) << "========== supress negative labels";
  // subsample negative labels if we have too many
  int num_bg = FrcnnParam::rpn_batchsize - get_equal_idx(labels, 1).size();
  vector<int> bg_inds = get_equal_idx(labels, 0);
  if (bg_inds.size() > num_bg) {
    std::set<int> ind_set;
    while (ind_set.size() < num_bg) {
      int tmp_idx = caffe::caffe_rng_rand() % bg_inds.size();
      ind_set.insert(bg_inds[tmp_idx]);
    }
    for (std::vector<int>::iterator it = bg_inds.begin(); it != bg_inds.end(); it++) {
      labels[*it] = -1;
    }
    for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end(); it++) {
      labels[*it] = 0;
    }
  }

  DLOG(ERROR) << "label == 1 : " << get_equal_idx(labels, 1).size();
  DLOG(ERROR) << "label == 0 : " << get_equal_idx(labels, 0).size();
  DLOG(ERROR) << "label == -1: " << get_equal_idx(labels, -1).size();

  DLOG(ERROR) << "========== transfer bbox";
  vector<Point4f<Dtype> > bbox_targets;
  if (gt_boxes.size() > 0) {
    vector<Point4f<Dtype> > max_overlap_gt_boxes;
    for (int i =0; i < argmax_overlaps.size(); i++) {
      max_overlap_gt_boxes.push_back(gt_boxes[argmax_overlaps[i]]);
    }
    bbox_targets = bbox_transform(anchors, max_overlap_gt_boxes);
  } else {
    bbox_targets = vector<Point4f<Dtype> >(n_anchors, Point4f<Dtype>());
  }

  vector<Point4f<Dtype> > bbox_inside_weights(n_anchors);
  for (int i = 0; i < n_anchors; i++) {
    if (labels[i] == 1) {
      //memcpy(bbox_inside_weights[i].Point, &FrcnnParam::rpn_bbox_inside_weights[0],
      //       4 * sizeof(float));
      bbox_inside_weights[i].Point[0] = FrcnnParam::rpn_bbox_inside_weights[0];
      bbox_inside_weights[i].Point[1] = FrcnnParam::rpn_bbox_inside_weights[1];
      bbox_inside_weights[i].Point[2] = FrcnnParam::rpn_bbox_inside_weights[2];
      bbox_inside_weights[i].Point[3] = FrcnnParam::rpn_bbox_inside_weights[3];
    }
  }

  Dtype bbox_outside_weight =  1.0 / (get_equal_idx(labels, 0).size()
                                      + get_equal_idx(labels, 1).size());
  //vector<Dtype> bbox_outside_weight_array(4, bbox_outside_weight);
  vector<Point4f<Dtype> > bbox_outside_weights(n_anchors);
  for (int i = 0; i < n_anchors; i++) {
    if (labels[i] >= 0) {
      //memcpy(bbox_outside_weights[i].Point, &bbox_outside_weight_array[0], 4 * sizeof(float));
      //bbox_outside_weights[i].Point[0] = bbox_outside_weight_array[0];
      //bbox_outside_weights[i].Point[1] = bbox_outside_weight_array[1];
      //bbox_outside_weights[i].Point[2] = bbox_outside_weight_array[2];
      //bbox_outside_weights[i].Point[3] = bbox_outside_weight_array[3];
      bbox_outside_weights[i].Point[0] = bbox_outside_weight;
      bbox_outside_weights[i].Point[1] = bbox_outside_weight;
      bbox_outside_weights[i].Point[2] = bbox_outside_weight;
      bbox_outside_weights[i].Point[3] = bbox_outside_weight;
    }
  }

  DLOG(ERROR) << "========== copy to top";
  // labels
  top[0]->Reshape(1, 1, config_n_anchors_ * height, width);
  Dtype *top_data = top[0]->mutable_cpu_data();
  Dtype fill = -1;
  unmap(labels, top_data, inds_inside, fill, config_n_anchors_, height, width);

  fill = 0;
  // bbox_targets
  top[1]->Reshape(1, config_n_anchors_ * 4, height, width);
  top_data = top[1]->mutable_cpu_data();
  unmap(bbox_targets, top_data, inds_inside, fill, 4 * config_n_anchors_, height, width);

  // bbox_inside_weights
  top[2]->Reshape(1, config_n_anchors_ * 4, height, width);
  top_data = top[2]->mutable_cpu_data();
  unmap(bbox_inside_weights, top_data, inds_inside, fill, 4 * config_n_anchors_, height, width);

  // bbox_outside_weights
  top[3]->Reshape(1, config_n_anchors_ * 4, height, width);
  top_data = top[3]->mutable_cpu_data();
  unmap(bbox_outside_weights, top_data, inds_inside, fill, 4 * config_n_anchors_, height, width);
}

template <typename Dtype>
void FrcnnAnchorTargetLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FrcnnAnchorTargetLayer);
#endif

INSTANTIATE_CLASS(FrcnnAnchorTargetLayer);
REGISTER_LAYER_CLASS(FrcnnAnchorTarget);

} // namespace frcnn

} // namespace caffe
