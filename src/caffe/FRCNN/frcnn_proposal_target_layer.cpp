// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/FRCNN/frcnn_proposal_target_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnProposalTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top) {
  config_n_classes_ = FrcnnParam::n_classes;

  LOG(INFO) << "FrcnnProposalTargetLayer :: " << config_n_classes_ << " classes";
  LOG(INFO) << "FrcnnProposalTargetLayer :: LayerSetUp";
  // sampled rois (0, x1, y1, x2, y2)
  top[0]->Reshape(1, 5, 1, 1);
  // labels
  top[1]->Reshape(1, 1, 1, 1);
  // bbox_targets
  top[2]->Reshape(1, config_n_classes_ * 4, 1, 1);
  // bbox_inside_weights
  top[3]->Reshape(1, config_n_classes_ * 4, 1, 1);
  // bbox_outside_weights
  top[4]->Reshape(1, config_n_classes_ * 4, 1, 1);
}

template <typename Dtype>
void put_bbox_regression_label(Dtype * data, const vector<Point4f<Dtype> >& target_data,
                               const vector<int>& labels, int num_classes) {
  // Bounding-box regression targets (bbox_target_data) are stored in a
  // compact form N x (class, tx, ty, tw, th)

  // This function expands those targets into the 4-of-4*K representation used
  // by the network (i.e. only one class has non-zero targets).

  // Returns:
  //     bbox_target (ndarray): N x 4K blob of regression targets
  //     bbox_inside_weights (ndarray): N x 4K blob of loss weights

  for (int i = 0; i < labels.size(); i++) {
    int cls = labels[i];
    if (cls > 0) {
      int start = 4 * cls + 4 * i * num_classes;
      for (int j = 0; j < 4; j++) {
        data[start + j] = target_data[i][j];
      }
    }
  }
}

template <typename Dtype>
void FrcnnProposalTargetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_rois = bottom[0]->cpu_data();
  const Dtype *bottom_gt_boxes = bottom[1]->cpu_data();

  vector<Point4f<Dtype> > rois;
  for (int i = 0; i < bottom[0]->num(); i++) {
    const Dtype * base_address = &bottom_rois[(i * bottom[0]->channels())];
    rois.push_back(Point4f<Dtype>(base_address[1], base_address[2], base_address[3],
                        base_address[4]));
  }

  vector<Point4f<Dtype> > gt_boxes;
  vector<int> gt_labels;
  for (int i = 0; i < bottom[1]->num(); i++) {
    const Dtype * base_address = &bottom_gt_boxes[(i * bottom[1]->channels())];
    gt_boxes.push_back(Point4f<Dtype>(base_address[0], base_address[1], base_address[2],
                            base_address[3]));
    gt_labels.push_back(base_address[4]);
  }

  rois.insert(rois.end(), gt_boxes.begin(), gt_boxes.end());

  const int n_rois = rois.size();

  // label: 1 is positive, 0 is negative, -1 is dont care
  vector<int> labels(n_rois, 0);

  vector<Dtype> max_overlaps(n_rois, 0);
  vector<int> argmax_overlaps(n_rois, -1);

  DLOG(ERROR) << "gt boxes size: " << gt_boxes.size();

  vector<vector<Dtype> > ious = get_ious(rois, gt_boxes);
  for (int i = 0; i < n_rois; i++) {
    for (size_t j = 0; j < gt_boxes.size(); j++) {
      if (ious[i][j] >= max_overlaps[i]) {
        max_overlaps[i] = ious[i][j];
        argmax_overlaps[i] = j;
      }
    }
    if (argmax_overlaps[i] >= 0) {
      labels[i] = gt_labels[argmax_overlaps[i]];
    }
  }

  vector<int> fg_inds, bg_inds;
  for (int i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= FrcnnParam::fg_thresh) {
      fg_inds.push_back(i);
    }
    if (max_overlaps[i] >= FrcnnParam::bg_thresh_lo &&
        max_overlaps[i] < FrcnnParam::bg_thresh_hi) {
      bg_inds.push_back(i);
    }
  }

  // subsample positive labels if we have too many
  int roi_per_image = FrcnnParam::batch_size;
  int num_fg = FrcnnParam::fg_fraction * FrcnnParam::batch_size;

  if (fg_inds.size() > num_fg) {
    std::set<int> ind_set;
    while (ind_set.size() < num_fg) {
      int tmp_idx = caffe::caffe_rng_rand() % fg_inds.size();
      ind_set.insert(fg_inds[tmp_idx]);
    }
    fg_inds.clear();
    for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end(); it++) {
      fg_inds.push_back(*it);
    }
  }
  num_fg = std::min(num_fg, static_cast<int>(fg_inds.size()));

  DLOG(ERROR) << "num_fg: " << num_fg;

  // subsample negative labels if we have too many
  int num_bg = roi_per_image - num_fg;
  if (bg_inds.size() > num_bg) {
    std::set<int> ind_set;
    while (ind_set.size() < num_bg) {
      int tmp_idx = caffe::caffe_rng_rand() % bg_inds.size();
      ind_set.insert(bg_inds[tmp_idx]);
    }
    bg_inds.clear();
    for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end(); it++) {
      bg_inds.push_back(*it);
    }
  }

  DLOG(ERROR) << "num_bg: " << num_bg;

  vector<int> keep_inds(fg_inds);
  keep_inds.insert(keep_inds.end(), bg_inds.begin(), bg_inds.end());

  vector<Point4f<Dtype> > sel_rois;
  vector<int> sel_labels;
  for (int i = 0; i < keep_inds.size(); i++) {
    sel_rois.push_back(rois[keep_inds[i]]);
    sel_labels.push_back(labels[keep_inds[i]]);
    if (i >= fg_inds.size()) {
      sel_labels[i] = 0;
    }
  }

  const int n_sel_rois = sel_rois.size();
  vector<Point4f<Dtype> > bbox_targets;
  if (gt_boxes.size() > 0) {
    vector<Point4f<Dtype> > max_overlap_gt_boxes;
    for (int i =0; i < n_sel_rois; i++) {
      max_overlap_gt_boxes.push_back(gt_boxes[argmax_overlaps[keep_inds[i]]]);
    }
    bbox_targets = bbox_transform(sel_rois, max_overlap_gt_boxes);
  } else {
    bbox_targets = vector<Point4f<Dtype> >(n_sel_rois, Point4f<Dtype>());
  }

  vector<Point4f<Dtype> > bbox_inside_weights(n_rois);
  for (int i = 0; i < n_sel_rois; i++) {
    if (labels[i] > 0) {
      bbox_inside_weights[i].Point[0] = FrcnnParam::rpn_bbox_inside_weights[0];
      bbox_inside_weights[i].Point[1] = FrcnnParam::rpn_bbox_inside_weights[1];
      bbox_inside_weights[i].Point[2] = FrcnnParam::rpn_bbox_inside_weights[2];
      bbox_inside_weights[i].Point[3] = FrcnnParam::rpn_bbox_inside_weights[3];
    }
  }

  DLOG(ERROR) << "top[0]-> " << n_sel_rois << " , 5, 1, 1";
  // sampled rois
  top[0]->Reshape(n_sel_rois, 5, 1, 1);
  Dtype *top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < n_sel_rois; i++) {
    Point4f<Dtype> &rect = sel_rois[i];
    top_data[i * 5] = 0;
    for (int j = 1; j < 5; j++) {
      top_data[i * 5 + j] = rect[j - 1];
    }
  }

  // classification labels
  top[1]->Reshape(n_sel_rois, 1, 1, 1);
  top_data = top[1]->mutable_cpu_data();
  for (int i = 0; i < n_sel_rois; i++) {
    top_data[i] = sel_labels[i];
  }

  // bbox_targets
  top[2]->Reshape(n_sel_rois, config_n_classes_ * 4, 1, 1);
  top_data = top[2]->mutable_cpu_data();
  put_bbox_regression_label(top_data, bbox_targets, sel_labels, config_n_classes_);

  // bbox_inside_weights
  top[3]->Reshape(n_sel_rois, config_n_classes_ * 4, 1, 1);
  top_data = top[3]->mutable_cpu_data();
  put_bbox_regression_label(top_data, bbox_inside_weights, sel_labels, config_n_classes_);

  // bbox_outside_weights
  top[4]->Reshape(n_sel_rois, config_n_classes_ * 4, 1, 1);
  top_data = top[4]->mutable_cpu_data();
  put_bbox_regression_label(top_data, bbox_inside_weights, sel_labels, config_n_classes_);
}

template <typename Dtype>
void FrcnnProposalTargetLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FrcnnProposalTargetLayer);
#endif

INSTANTIATE_CLASS(FrcnnProposalTargetLayer);
REGISTER_LAYER_CLASS(FrcnnProposalTarget);

} // namespace frcnn

} // namespace caffe
