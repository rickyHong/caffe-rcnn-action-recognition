// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/FRCNN/frcnn_rpn_recall_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnRpnRecallLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(top.size(), 1);
  //CHECK_EQ(bottom[0].num(), bottom[1].num());
  //CHECK_EQ(bottom[0].channels(), bottom[1].channels());
  //CHECK_EQ(bottom[0].height(), bottom[1].height());
  //CHECK_EQ(bottom[0].width(), bottom[1].width());
  const int divide_num = 10;
  overlap.resize(divide_num);
  recalled.resize(divide_num);
  for (int i = 0; i < divide_num; ++i) {
    overlap[i] = i * 1. / divide_num;
    recalled[i] = 0;
  }
  top[0]->Reshape(divide_num, 1, 1, 1);
  this->total_count = 0;
}

template <typename Dtype>
void FrcnnRpnRecallLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_gt_bbox = bottom[1]->cpu_data();
  const Dtype *bottom_rois = bottom[0]->cpu_data(); 
  vector<Point4f<Dtype> > gt_boxes;
  for (int i = 0; i < bottom[1]->num(); i++) {
    const Dtype * base_address = &bottom_gt_bbox[(i * bottom[1]->channels())];
    gt_boxes.push_back(Point4f<Dtype>(base_address[0], base_address[1], base_address[2],base_address[3]));  
  } 
  vector<Point4f<Dtype> > rois;
  for (int i = 0; i < bottom[0]->num(); i++) {
    const Dtype * base_address = &bottom_rois[(i * bottom[0]->channels())];
    rois.push_back(Point4f<Dtype>(base_address[1], base_address[2], base_address[3],base_address[4]));
  }
  this->total_count += gt_boxes.size();
  
  for (size_t over_id = 0; over_id < overlap.size(); over_id++) {
    const float overlap_ = overlap[over_id];

    vector<bool> used(rois.size(), false);
    for (size_t i = 0; i < gt_boxes.size(); ++i) {
      size_t idx = -1;
      Dtype mi_over = 0;
      for (size_t rois_id = 0; rois_id < rois.size(); ++rois_id) {
        Dtype current_overlap = get_iou(rois[rois_id],gt_boxes[i]);
        if (used[rois_id] == false && current_overlap >= overlap_) {
          if (idx == -1 || mi_over > current_overlap) {
            idx = rois_id;
            mi_over = current_overlap; 
          } 
        }
      } 
      if (idx != -1) {
        used[idx] = true;
        recalled[over_id] ++;
      }
    }
  }
  CHECK_EQ(top[0]->count(), overlap.size());
  for (int i = 0; i < top[0]->count(); ++i) {
    top[0]->mutable_cpu_data()[i] = Dtype(recalled[i]) / Dtype(this->total_count);
  }
}

template <typename Dtype>
void FrcnnRpnRecallLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FrcnnRpnRecallLayer);
#endif

INSTANTIATE_CLASS(FrcnnRpnRecallLayer);
REGISTER_LAYER_CLASS(FrcnnRpnRecall);

} // namespace frcnn

} // namespace caffe
