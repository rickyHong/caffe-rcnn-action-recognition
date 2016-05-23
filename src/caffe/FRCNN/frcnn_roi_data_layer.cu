#include "caffe/FRCNN/frcnn_roi_data_layer.hpp"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data, Image Blob
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // top[1] is image_info , top[2] is gx_bpxes
    caffe_copy(3, batch->label_.gpu_data(), top[1]->mutable_gpu_data());
    // Reshape to loaded labels.
    top[2]->Reshape(batch->label_.num()-1, batch->label_.channels(), batch->label_.height(), batch->label_.width());
    // Copy the labels.
    // First five is image_info
    caffe_copy(batch->label_.count() - 5, batch->label_.gpu_data() + 5, top[2]->mutable_gpu_data());

    // DEBUG INFORMATION
    DLOG(ERROR) << "[" << (this->phase_==TRAIN?"TRAIN":"TEST") << "] FrcnnRoiDataLayer Forward: " << image_database_[batch->label_.cpu_data()[3]] << " scale : " << top[1]->cpu_data()[2] << " [] GT_NUM : " << top[2]->num();
    DLOG(ERROR) << "FrcnnRoiDataLayer Forward: " << "height : " << top[1]->cpu_data()[0] << ", width : " << top[1]->cpu_data()[1] ;
    for (int i = 0; i < top[2]->num(); i++) {
      DLOG(ERROR) << "---------= " << top[2]->cpu_data()[5 * i + 0] << ", " << top[2]->cpu_data()[5 * i + 1] << ", " << top[2]->cpu_data()[5 * i + 2] << ", " << top[2]->cpu_data()[5 * i + 3] << "   . label : " << top[2]->cpu_data()[5 * i + 4];
    }
    DLOG(ERROR) << "FrcnnRoiDataLayer Forward >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> END";
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(FrcnnRoiDataLayer);

}  // namespace Frcnn

}  // namespace caffe
