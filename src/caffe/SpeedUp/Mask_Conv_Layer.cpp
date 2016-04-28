#include <iostream>
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/SpeedUp/Mask_Conv_Layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
#define CUDNN_STREAMS_PER_GROUP 3
#endif

inline bool Is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Mask case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  //group_ = this->layer_param_.convolution_param().group();
  this->group_ = 1;
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // ==========================================================================
  // Initialize CUDA streams and cuDNN.
#ifdef USE_CUDNN
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* cudnn_kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = cudnn_kernel_shape_data[0];
  const int kernel_w = cudnn_kernel_shape_data[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
#endif

  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[1]->channels(), this->group_);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  //CHECK_EQ(top[0]->channels()%this->group_, 0);
  CHECK_EQ(top.size(), 1);
  group_no_zero.resize(this->group_);
  col_buffer_shape_.resize(this->group_);
  col_buffer_.resize(this->group_);
  bias_multiplier_.resize(this->group_);
  buffer_stride_.resize(this->group_);
  c_top_.resize(this->group_);
  for (int index = 0; index < this->group_; index++) {
    col_buffer_[index].reset(new Blob<Dtype>());
    bias_multiplier_[index].reset(new Blob<Dtype>());
    buffer_stride_[index].reset(new Blob<int>());
    c_top_[index].reset(new Blob<Dtype>());
  }
  
  LOG(INFO) << this->layer_param().name() << " >>>>>> ";
  LOG(INFO) << "kernel_dim_       : " << kernel_dim_;
  LOG(INFO) << "num_spatial_axes_ : " << num_spatial_axes_;
  num_ = bottom[0]->count(0, channel_axis_);
  LOG(INFO) << "channel_axis_     : " << channel_axis_;
  LOG(INFO) << "num_              : " << num_;
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  CHECK_EQ(dilation_.cpu_data()[0], 1);
  CHECK_EQ(dilation_.cpu_data()[1], 1);
  LOG(INFO) << "dilation_ [" << dilation_.count() << "]  " << dilation_.cpu_data()[0] << ", " << dilation_.cpu_data()[1];
  for (int index = 0; index < num_spatial_axes_; index++) {
    LOG(INFO) << "output_shape[" << index << "] = " << output_shape_[index];
  }

  Reshape(bottom, top);
  LOG(INFO) << "conv_out_channels_: " << conv_out_channels_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  LOG(INFO) << "conv_out_spatial_dim_ : " << conv_out_spatial_dim_;
  LOG(INFO) << "weight_offset_    : " << weight_offset_;
  LOG(INFO) << "output_offset_    : " << output_offset_;
  LOG(INFO) << "bottom_dim_       : " << this->bottom_dim_;
  CHECK_EQ(force_nd_im2col_, false);      // for conv_im2col_cpu use im2col_cpu
  CHECK_EQ(num_spatial_axes_, 2);
} 

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::PrepareMask(const int num_index, const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[1]->channels(), this->group_);
  const int height = top[0]->height();
  const int width = top[0]->width();
  const int spatial_dim = height * width;
  DLOG(INFO) << "bottom shape : " << bottom[1]->num() << ", " << bottom[1]->channels() << ", " << bottom[1]->height() << ", " << bottom[1]->width();
  DLOG(INFO) << "num_index : " << num_index << ", spatial_dim : " << spatial_dim;
  DLOG(INFO) << "mask shift : " << bottom[1]->offset(num_index) << "  c_top_.size : " << c_top_.size();
  for (int index = 0; index < this->group_; index++) {
    //const Dtype* mask = bottom[1]->cpu_data() + bottom[1]->offset(num_index, index);
    DLOG(INFO) << "c_top_[" << index << "]  Reshape " << 1 << ", " << top[0]->channels() / this->group_ << ", " << top[0]->height() << ", " << top[0]->width();

    c_top_[index]->Reshape(1, top[0]->channels() / this->group_, top[0]->height(), top[0]->width());
    group_no_zero[index] = 0;
    DLOG(INFO) << "buffer_stride_[" << index << "] Reshape " << "1, 1, " << height << ", " << width;
    buffer_stride_[index]->Reshape(1, 1, height, width);
    int last = 1;
    int* current_buffer_stride_ = buffer_stride_[index]->mutable_cpu_data();
    for (int idx = spatial_dim-1; idx >= 0; idx--) {
      current_buffer_stride_[idx] = last;
      //if (mask[idx] > 0 || idx == 0) {
      if (idx % 2 == 0){
        last = 1;
        group_no_zero[index]++;
        if (idx == 0) {
          bottom[1]->mutable_cpu_data()[bottom[1]->offset(num_index, index)+idx] = 1;
        }
      }else {
        last ++;
      }
    }
    DLOG(INFO) << index << " / " << this->group_
        << " Sparsity : " << group_no_zero[index] << " / " << buffer_stride_[index]->count();
    if (group_no_zero[index] == 0) continue;
    // Col_buffer
    col_buffer_shape_[index].clear();
    col_buffer_shape_[index].push_back(kernel_dim_);
    col_buffer_shape_[index].push_back(group_no_zero[index]);
    /*
    for (int i = 0; i < num_spatial_axes_; ++i) {
      col_buffer_shape_[index].push_back(output_shape_[i]);
    }*/
    col_buffer_[index]->Reshape(col_buffer_shape_[index]);
    DLOG(INFO) << this->layer_param().name() << " ====== " << " col_buffer_[" << index << "] " << col_buffer_[index]->num() << ", " << col_buffer_[index]->channels() << ", " << col_buffer_[index]->height() << ", " << col_buffer_[index]->width();

    // bias_multiplier_
    vector<int> bias_multiplier_shape(1, group_no_zero[index]);
    bias_multiplier_[index]->Reshape(bias_multiplier_shape);
    DLOG(INFO) << this->layer_param().name() << " ====== " << " bias_multiplier_[" << index << "] " << bias_multiplier_[index]->num() << ", " << bias_multiplier_[index]->channels() << ", " << bias_multiplier_[index]->height() << ", " << bias_multiplier_[index]->width();
    caffe_set(bias_multiplier_[index]->count(), Dtype(1),
      bias_multiplier_[index]->mutable_cpu_data());
  }
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::base_im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col, const shared_ptr<Blob<int> > stride_blob,const int limit) {
  //const int output_h = (height + 2 * pad_h - kernel_h ) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - kernel_w ) / stride_w + 1;
  const int channel_size = height * width;
  //CHECK_EQ(output_h, stride_blob->height());
  //CHECK_EQ(output_w, stride_blob->width());
  //CHECK_EQ(stride_blob->num(), 1);
  //CHECK_EQ(stride_blob->channels(), 1);
  //DLOG(INFO) << "<> base_im2col_cpu : " << output_h << ", " << output_w << ", " << width ;
  const int *STRIDE = stride_blob->cpu_data();
  const int count = stride_blob->count();
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        //int input_row = -pad_h + kernel_row ;
        //int debug = 0;
        for (int index = 0; index < count; 
            index+=STRIDE[index] ){
          int input_row = -pad_h + kernel_row + stride_h * (index / output_w);
          int input_col = -pad_w + kernel_col + stride_w * (index % output_w);
          //DLOG(INFO) << "input channel row col : " << channels-channel-1 << ", " << input_row << ", " << input_col;
          //CHECK_LT(debug, limit) << debug << " / " << group_no_zero[0]
          //  << "  {}  " << channels-channel-1 << ", " << input_row << ", " << input_col;
          //if (!Is_a_ge_zero_and_a_lt_b(input_row, height) || !Is_a_ge_zero_and_a_lt_b(input_col, width)){
        if (true){
            *(data_col++) = 0;
          } else {
            *(data_col++) = data_im[input_row * width + input_col];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
/*
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
*/
  CHECK_EQ(bottom.size(), 2) << "Only Support Two Input Blobs, One is Feature Map, The Othrer is Mask";
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  //col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  //output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ ;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  //col_buffer_shape_.clear();
  //col_buffer_shape_.push_back(kernel_dim_ );
  //for (int i = 0; i < num_spatial_axes_; ++i) {
  //  col_buffer_shape_.push_back(output_shape_[i]);
  //}
  //col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
/*
  if (bias_term_) {
    LOG(INFO) << this->layer_param().name() << " bias_term | out_spatial_dim: " << out_spatial_dim_;
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
*/
#ifdef USE_CUDNN
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
  bottom_offset_ = this->bottom_dim_ ;
  top_offset_ = this->top_dim_ ;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ , height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ , height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);

    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // choose backward algorithm for filter
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          workspace_limit_bytes, &bwd_filter_algo_[i]) );

    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // choose backward algo for data
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_[i]));

    // get workspace size
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (1/*this->group_*/ * CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (1/*this->group_*/ * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (1/*this->group_*/ * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ /* /this->group_*/, 1, 1);
  }
#endif
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights ) {
  CHECK_EQ(c_top_.size(), this->group_);
  for (int g = 0; g < group_; ++g) {
    CHECK_EQ(conv_out_channels_ / group_, c_top_[g]->channels());

    LOG(INFO) << "forward_cpu_gemm pre conv_im2col_cpu";
    conv_im2col_cpu(input, col_buffer_[g]->mutable_cpu_data(), buffer_stride_[g], group_no_zero[g]);

    LOG(INFO) << " forward_cpu_gemm " << g << " / " << this->group_ << " = conv_im2col_cpu";
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
      group_, group_no_zero[g], kernel_dim_,
      (Dtype)1., weights + weight_offset_ * g, col_buffer_[g]->cpu_data(),
      (Dtype)0., c_top_[g]->mutable_cpu_data());

    LOG(INFO) << " forward_cpu_gemm "
        << g << " / " << this->group_ << " = caffe_cpu_gemm";
  }
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::forward_cpu_bias(const Dtype* bias) {
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      group_no_zero[g], 1, (Dtype)1., bias, bias_multiplier_[g]->cpu_data(),
      (Dtype)1., c_top_[g]->mutable_cpu_data());
  }
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::Recover(Dtype* top_data){
  for (int g = 0; g < group_; ++g) {
    const Dtype* data = c_top_[g]->cpu_data();
    for (int index = 0, ii = 0; index < buffer_stride_[g]->count();
        index+=buffer_stride_[g]->cpu_data()[index], ii++){
      for (int _c = 0; _c < conv_out_channels_ / group_; _c++) {
        top_data[_c*conv_out_spatial_dim_ + index] = data[ii + _c*group_no_zero[g]];
        const int shift = buffer_stride_[g]->cpu_data()[index] - 1;
        if (shift>0) caffe_set(shift, Dtype(0), top_data + _c*conv_out_spatial_dim_ + index + 1);  
      }
    }
    top_data += conv_out_channels_ / group_ * conv_out_spatial_dim_;
  }
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  CHECK_EQ(this->num_, bottom[0]->num());
  for (int n = 0; n < this->num_; ++n) {
    LOG(INFO) << ">>>>> " << this->layer_param().name() ;
    this->PrepareMask(n, bottom, top);
    LOG(INFO) << ">>>>> " << this->layer_param().name() << " PrepareMask";
    this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight);
    LOG(INFO) << ">>>>> " << this->layer_param().name() << " Forward_CPU_Gemm";
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->cpu_data();
      this->forward_cpu_bias(bias);
      LOG(INFO) << ">>>>> " << this->layer_param().name() << " Forward_CPU_Bias";
    }
    Recover(top_data + n * this->top_dim_);
    LOG(INFO) << ">>>>> " << this->layer_param().name() << " Recover";
  }
}

template <typename Dtype>
void MaskConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //NOT_IMPLEMENTED;
}

#ifdef USE_CUDNN
template <typename Dtype>
MaskConvolutionLayer<Dtype>::~MaskConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < 1/*this->group_*/ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  cudaFree(workspaceData);
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}
#endif

#ifdef CPU_ONLY
STUB_GPU(MaskConvolutionLayer);
#endif 

INSTANTIATE_CLASS(MaskConvolutionLayer);
REGISTER_LAYER_CLASS(MaskConvolution);

}  // namespace caffe
