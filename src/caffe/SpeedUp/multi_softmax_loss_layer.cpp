#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/SpeedUp/multi_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
int CalculateSparsity__(const Blob<Dtype>* data, Dtype thresh, const int channel_stride) {
  CHECK_EQ(data->channels()%channel_stride, 0);
  int zero = 0;
  for (int index = 0; index < data->num(); index++) {
    for (int ic = 0; ic < data->channels(); ic+=channel_stride) {
      for (int ih = 0; ih < data->height(); ih++) {
        for (int iw = 0; iw < data->width(); iw++) {
          if (data->cpu_data()[data->offset(index,ic,ih,iw)] <= thresh) {
            zero ++;
          }
        }
      }
    }
  }
  return zero;
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  this->layer_iter_ = 0;
  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  
  const int predict_num = bottom[0]->channels();
  const int label_num = bottom[1]->channels();
  CHECK_EQ(predict_num, this->group_*2);
  CHECK_EQ(label_num % this->group_, 0);
  copy_for_softmax_num_ = label_num / this->group_;
  thresh_for_generate_softmax_label_ = this->layer_param().threshold_param().threshold();

  // Set up slice_layer_ to slice the bottom[0] into softmax input
  //const int num_for_slice = bottom[0]->num() * this->group_;
  //CHECK_EQ(num_for_slice*2, bottom[0]->num()*bottom[0]->channels());
  //bottom[0]->Reshape(num_for_slice, 2, bottom[0]->height(), bottom[0]->width());
  // Set up split_layer_ to copy bottom[0] into  softmax input
  LOG(INFO) << "MultiSoftmax Reshap bottom[0] : " << bottom[0]->num() << " , " << bottom[0]->channels() << " , " << bottom[0]->height() << " , " << bottom[0]->width();
  LOG(INFO) << "MultiSoftmax Reshap bottom[1] : " << bottom[1]->num() << " , " << bottom[1]->channels() << " , " << bottom[1]->height() << " , " << bottom[1]->width();
  LayerParameter reshape_param;
  ::caffe::ReshapeParameter* reshape_param_ = reshape_param.mutable_reshape_param();
  ::caffe::BlobShape* blob_shape_ = reshape_param_->mutable_shape();
  blob_shape_->clear_dim();
  blob_shape_->add_dim(-1);
  blob_shape_->add_dim(2);
  blob_shape_->add_dim(0);
  blob_shape_->add_dim(0);
  reshape_layer_.reset(new ReshapeLayer<Dtype>(reshape_param));
  reshape_top_vec_.clear();
  reshape_top_vec_.push_back(&reshape_top_);
  reshape_bottom_vec_.clear();
  reshape_bottom_vec_.push_back(bottom[0]);
  reshape_layer_->SetUp(reshape_bottom_vec_, reshape_top_vec_);
  reshape_layer_->Reshape(reshape_bottom_vec_, reshape_top_vec_);
  LOG(INFO) << "Reshap Layer top[0] : " << reshape_top_vec_[0]->num() << " , " << reshape_top_vec_[0]->channels() << " , " << reshape_top_vec_[0]->height() << " , " << reshape_top_vec_[0]->width();
  

  LayerParameter split_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_param));
  split_bottom_vec_.clear();
  split_top_vec_.clear();
  split_bottom_vec_.push_back(&reshape_top_);
  for (int index = 0; index < copy_for_softmax_num_; index++) {
    split_top_vec_.push_back(new Blob<Dtype>());
  }
  split_layer_->SetUp(split_bottom_vec_, split_top_vec_);
  LOG(INFO) << "SplitLayer, top has " << split_top_vec_.size() << " blobs";

  LOG(INFO) << "copy_for_softmax_num : " << copy_for_softmax_num_;
  //CHECK(this->layer_param_.loss_weight_size() == 1
  //      || this->layer_param_.loss_weight_size() == 0);
  //Dtype loss_weight = 1;
  //if (this->layer_param_.loss_weight_size() == 1) 
  //  loss_weight = this->layer_param_.loss_weight(0);
  //LOG(INFO) << "MultiSoftmax loss weight : " << loss_weight;
  accu_layer_.resize( copy_for_softmax_num_ );
  accu_bottom_vec_.resize( copy_for_softmax_num_ );
  accu_top_vec_.resize( copy_for_softmax_num_ );

  LOG(INFO) << "MultiSoftmax Layer has " << this->layer_param_.loss_weight_size() << " loss weights";
  CHECK_EQ(this->layer_param_.loss_weight_size(), 2);
  LayerParameter softmax_loss_param = this->layer_param();
  softmax_loss_param.clear_loss_weight();
  softmax_loss_param.add_loss_weight(this->layer_param_.loss_weight(0)/copy_for_softmax_num_);
  LOG(INFO) << "Softmax_Loss_Param has " << softmax_loss_param.loss_weight_size() << " loss weights";
  
  LayerParameter accu_param = this->layer_param();
  accu_param.clear_loss_weight();
  accu_param.add_loss_weight(Dtype(0));
  LOG(INFO) << "Accuracy_Param has " << accu_param.loss_weight_size() << " loss weights";

  softmax_loss_layer_.resize( copy_for_softmax_num_ );
  softmax_loss_top_vec_.resize( copy_for_softmax_num_ );
  softmax_loss_bottom_vec_.resize( copy_for_softmax_num_ );
  //CHECK_EQ(this->group_ * copy_for_softmax_num_, N * bottom[1]->channels());
  for (int index = 0; index < copy_for_softmax_num_; index++) {
    softmax_loss_layer_[index].reset(new SoftmaxWithLossLayer<Dtype>(softmax_loss_param));
    softmax_loss_top_vec_[index].clear();
    softmax_loss_top_vec_[index].push_back(new Blob<Dtype>());

    CHECK_LT(index, split_top_vec_.size());
    softmax_loss_bottom_vec_[index].clear();
    softmax_loss_bottom_vec_[index].push_back( split_top_vec_[index] ); 
    softmax_loss_bottom_vec_[index].push_back(new Blob<Dtype>(split_top_vec_[index]->num(), 1, split_top_vec_[index]->height(), split_top_vec_[index]->width()));
    softmax_loss_layer_[index]->SetUp(softmax_loss_bottom_vec_[index], softmax_loss_top_vec_[index]);
    LOG(INFO) << "MultiSoftmaxWithLoss LayerSetUp : generate softmax_loss layer " << index << " / " << copy_for_softmax_num_ << " with top[0]->cpu_diff()[0] = " << softmax_loss_top_vec_[index][0]->cpu_diff()[0];

    accu_layer_[index].reset(new AccuracyLayer<Dtype>(accu_param));
    accu_top_vec_[index].push_back(new Blob<Dtype>());
    accu_bottom_vec_[index].clear();
    accu_bottom_vec_[index].push_back(softmax_loss_bottom_vec_[index][0]);
    accu_bottom_vec_[index].push_back(softmax_loss_bottom_vec_[index][1]);
    accu_layer_[index]->SetUp(accu_bottom_vec_[index], accu_top_vec_[index]);
    //softmax_loss_top_vec_[index]->set_loss(0, this->layer_param()->loss_weight()/this->group_); 
    CHECK_EQ(softmax_loss_top_vec_[index][0]->count(), 1);
    //caffe_set(softmax_loss_top_vec_[index][0]->count(), loss_weight/this->group_
    //            , softmax_loss_top_vec_[index][0]->mutable_cpu_diff());
  }
  LOG(INFO) << "Softmax_Loss " << copy_for_softmax_num_ << " softmax loss layers";
  LOG(INFO) << "Softmax_Loss bottom[0] " << softmax_loss_bottom_vec_[0][0]->num() << " , " 
    <<  softmax_loss_bottom_vec_[0][0]->channels() << " , " 
    << softmax_loss_bottom_vec_[0][0]-> height() << " , " 
    << softmax_loss_bottom_vec_[0][0]->width();
  LOG(INFO) << "Softmax_Loss bottom[1] " <<  softmax_loss_bottom_vec_[0][1]->num() << " , " 
    <<  softmax_loss_bottom_vec_[0][1]->channels() << " , " 
    << softmax_loss_bottom_vec_[0][1]-> height() << " , " 
    << softmax_loss_bottom_vec_[0][1]->width();

  
  CHECK_EQ(top.size(), 2);
  top[0]->Reshape(1, 1, 1, 1);
  top[1]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  reshape_layer_->Reshape(reshape_bottom_vec_, reshape_top_vec_);
  CHECK_EQ(reshape_top_vec_[0]->height(), bottom[0]->height());
  CHECK_EQ(reshape_top_vec_[0]->width(), bottom[0]->width());
  CHECK_EQ(reshape_top_vec_[0]->channels(), 2); 
  //const int num_for_slice = bottom[0]->num() * this->group_;
  //CHECK_EQ(num_for_slice*2, bottom[0]->num()*bottom[0]->channels());
  //bottom[0]->Reshape(num_for_slice, 2, bottom[0]->height(), bottom[0]->width());

  split_layer_->Reshape(split_bottom_vec_, split_top_vec_);

  const int size_softmax = softmax_loss_layer_.size();
  int count_for_mask_label = 0;
  for (int index = 0; index < size_softmax; index++) {
    vector<int> shape = softmax_loss_bottom_vec_[index][0]->shape();
    shape[1] = 1;
    softmax_loss_bottom_vec_[index][1]->Reshape(shape);
    count_for_mask_label += softmax_loss_bottom_vec_[index][1]->count();
    softmax_loss_layer_[index]->Reshape(softmax_loss_bottom_vec_[index], softmax_loss_top_vec_[index]);
    accu_layer_[index]->Reshape(accu_bottom_vec_[index], accu_top_vec_[index]);
    
  }
  CHECK_EQ(count_for_mask_label, bottom[1]->count());
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  reshape_layer_->Forward(reshape_bottom_vec_, reshape_top_vec_);
  split_layer_->Forward(split_bottom_vec_, split_top_vec_);

  const int size_softmax = softmax_loss_layer_.size();
  const int height_ = bottom[1]->height();
  const int width_ = bottom[1]->width();
  const int spatial_dim_ = height_ * width_;
  Dtype total_loss = Dtype(0);
  const int num = bottom[1]->num();
  const int stride = bottom[1]->channels() / group_;
  CHECK_EQ(bottom[1]->channels() % group_, 0);
  CHECK_EQ(softmax_loss_bottom_vec_[0][0]->num() , num*group_);
  CHECK_EQ(stride, size_softmax);
  Dtype accuracy = 0;
  bool cal_accu = this->phase_ == TEST || layer_iter_ % this->layer_param().crop_param().offset(0) == 0 ; 
  
  for (int index = 0; index < size_softmax; index++) {
    CHECK_EQ(softmax_loss_bottom_vec_[index][1]->count()*2, softmax_loss_bottom_vec_[index][0]->count());
    for (int in = 0; in < num; in++) {
      for (int ic = 0; ic < group_; ic++) {
        const int offset_top = in*group_+ic;
        const int offset_btm = bottom[1]->offset(in, ic*stride+index);
        const Dtype* bottom_data = bottom[1]->cpu_data() + offset_btm;
        Dtype* top_data = softmax_loss_bottom_vec_[index][1]->mutable_cpu_data() + offset_top*spatial_dim_;
        for (int jj = 0; jj < spatial_dim_; jj++){
          top_data[jj] = bottom_data[jj] <= thresh_for_generate_softmax_label_ ? Dtype(0) : Dtype(1);
        }
      }
    }
    softmax_loss_layer_[index]->Forward(softmax_loss_bottom_vec_[index], softmax_loss_top_vec_[index]);
    CHECK_EQ(softmax_loss_top_vec_[index].size(), 1);
    total_loss += softmax_loss_top_vec_[index][0]->cpu_data()[0];
    if (cal_accu) {
      accu_layer_[index]->Forward(accu_bottom_vec_[index], accu_top_vec_[index]);
      accuracy += accu_top_vec_[index][0]->cpu_data()[0] / size_softmax;
    }
  }
  LOG_EVERY_N(INFO, 1000) << ">>> value >>> " << this->layer_param_.name() << "  " << CalculateSparsity__(bottom[1], thresh_for_generate_softmax_label_, 1) << " / " << bottom[1]->count() << "  = " << CalculateSparsity__(bottom[1], thresh_for_generate_softmax_label_, 1)*1./bottom[1]->count();
  LOG_EVERY_N(INFO, 1000) << ">>> mask  >>> " << this->layer_param_.name() << "  " << CalculateSparsity__(bottom[0], thresh_for_generate_softmax_label_, 2) << " / " << bottom[0]->count()/2 << "  = " << CalculateSparsity__(bottom[0], thresh_for_generate_softmax_label_, 2)*2./bottom[0]->count();
  CHECK_EQ(top.size(), 2);
  CHECK_EQ(top[0]->count(), 1);
  top[0]->mutable_cpu_data()[0] = total_loss;
  if (cal_accu) top[1]->mutable_cpu_data()[0] = accuracy;
  layer_iter_ ++;
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int size_softmax = softmax_loss_layer_.size();
    for (int index = 0; index < size_softmax; index++) {
      softmax_loss_layer_[index]->Backward(softmax_loss_top_vec_[index], propagate_down, softmax_loss_bottom_vec_[index]);
    }

    const vector<bool> split_propagate_down(1, true);
    split_layer_->Backward(split_top_vec_, split_propagate_down, split_bottom_vec_);
    
    reshape_layer_->Backward(reshape_top_vec_, split_propagate_down, reshape_bottom_vec_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(MultiSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(MultiSoftmaxWithLoss);

}  // namespace caffe
