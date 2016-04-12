// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/31
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_PROPOSAL_TARGET_LAYER_HPP_
#define CAFFE_FRCNN_PROPOSAL_TARGET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

namespace Frcnn {

/*************************************************
FRCNN_PROPOSAL_TARGET
Assign object detection proposals to ground-truth targets. Produces proposal
classification labels and bounding-box regression targets.
bottom: 'rpn_rois'
bottom: 'gt_boxes'
top: 'rois'
top: 'labels'
top: 'bbox_targets'
top: 'bbox_inside_weights'
top: 'bbox_outside_weights'
**************************************************/
template <typename Dtype>
class FrcnnProposalTargetLayer : public Layer<Dtype> {
 public:
  explicit FrcnnProposalTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){}

  virtual inline const char* type() const { return "FrcnnProposalTarget"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 5; }
  virtual inline int MaxTopBlobs() const { return 5; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  vector<float> anchors_;
  int config_n_anchors_;
  int config_n_classes_;
};

}  // namespace frcnn

}  // namespace caffe

#endif  // CAFFE_FRCNN_PROPOSAL_TARGET_LAYER_HPP_
