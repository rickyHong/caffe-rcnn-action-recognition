#ifndef CAFFE_SPECIAL_CONVOLUTION_LAYER_HPP_
#define CAFFE_SPECIAL_CONVOLUTION_LAYER_HPP_

#include <vector>
#include <string>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

/**
 * Based on the Convolution Layer ,
 * -- Compute Time For All Parts in Convolution Layer.
 * -- 
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 * Change For Mask
 * First Is Original Blob
 * Second Mask Output Blob
 */
template <typename Dtype>
class MaskConvolutionLayer : public Layer<Dtype> {
 public:
#ifdef USE_CUDNN
  explicit MaskConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param), handles_setup_(false) {}
#else
  explicit MaskConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
#endif
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
#ifdef USE_CUDNN
  virtual ~MaskConvolutionLayer();
#endif

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "MaskConvolution"; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape(); 

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  //vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff
        , const shared_ptr<Blob<int> > stride_blob, const int limit) {
    DLOG(INFO) << "conv_im2col_cpu function : " << std::endl 
        << "conv_in_channels_　" << conv_in_channels_ << std::endl
        << "conv_input_shape_  " << conv_input_shape_.cpu_data()[1] << ", " << conv_input_shape_.cpu_data()[2] << std::endl
        << "kernel_shape_      " << kernel_shape_.cpu_data()[0] << ", " << kernel_shape_.cpu_data()[1] << std::endl
        << "pad_               " << pad_.cpu_data()[0] << ", " << pad_.cpu_data()[1] << std::endl
        << "stride_            " << stride_.cpu_data()[0] << ", " << stride_.cpu_data()[1];
    base_im2col_cpu(data, conv_in_channels_,
        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
        pad_.cpu_data()[0], pad_.cpu_data()[1],
        stride_.cpu_data()[0], stride_.cpu_data()[1],
        col_buff, stride_blob, limit);
  }

  void base_im2col_cpu(const Dtype* data_im, const int channels, 
        const int height, const int width, const int kernel_h, const int kernel_w, 
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        Dtype* data_col, const shared_ptr<Blob<int> > stride_blob, const int limit);

  void PrepareMask(const int num_index, const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights);
  void forward_cpu_bias(const Dtype* bias);
  void Recover(Dtype* top_data);

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  //Blob<Dtype> col_buffer_;
  //Blob<Dtype> bias_multiplier_;

// For Mask
  vector<shared_ptr<Blob<Dtype> > > col_buffer_;
  vector<shared_ptr<Blob<Dtype> > > bias_multiplier_;
  //@brief The spatial dimensions of the col_buffer.
  vector<vector<int> > col_buffer_shape_;
  vector<shared_ptr<Blob<int> > > buffer_stride_; // Get The Next no-zero value
  vector<int> group_no_zero;
  vector<shared_ptr<Blob<Dtype> > > c_top_;
// For CUDNN 
#ifdef USE_CUDNN
  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t*  stream_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t *fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t *workspace_bwd_filter_sizes_;
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData
#endif
};

}  // namespace caffe

#endif  // CAFFE_SPECIAL_CONVOLUTION_LAYER_HPP_
