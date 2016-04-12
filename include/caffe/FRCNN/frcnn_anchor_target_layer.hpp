// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/30
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_ANCHOR_TARGET_LAYER_HPP_
#define CAFFE_FRCNN_ANCHOR_TARGET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

namespace Frcnn {

/*************************************************
FRCNN_ANCHOR_TARGET
Assign anchors to ground-truth targets. Produces anchor classification
labels and bounding-box regression targets.
bottom: 'rpn_cls_score'
bottom: 'gt_boxes'
bottom: 'im_info'
top: 'rpn_labels'
top: 'rpn_bbox_targets'
top: 'rpn_bbox_inside_weights'
top: 'rpn_bbox_outside_weights'
**************************************************/
template <typename Dtype>
class FrcnnAnchorTargetLayer : public Layer<Dtype> {
 public:
  explicit FrcnnAnchorTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){};

  virtual inline const char* type() const { return "FrcnnAnchorTarget"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 4; }
  virtual inline int MaxTopBlobs() const { return 4; }

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
  int feat_stride_;
  float border_;
};

}  // namespace Frcnn

}  // namespace caffe

#endif  // CAFFE_FRCNN_ANCHOR_TARGET_LAYER_HPP_
