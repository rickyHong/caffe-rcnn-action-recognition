// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/31
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_PROPOSAL_LAYER_HPP_
#define CAFFE_FRCNN_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

namespace Frcnn {

/*************************************************
FrcnnProposalLayer
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
bottom: 'rpn_cls_prob_reshape'
bottom: 'rpn_bbox_pred'
bottom: 'im_info'
top: 'rpn_rois'
**************************************************/
template <typename Dtype>
class FrcnnProposalLayer : public Layer<Dtype> {
 public:
  explicit FrcnnProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){};

  virtual inline const char* type() const { return "FrcnnProposal"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace frcnn

}  // namespace caffe

#endif  // CAFFE_FRCNN_PROPOSAL_LAYER_HPP_
