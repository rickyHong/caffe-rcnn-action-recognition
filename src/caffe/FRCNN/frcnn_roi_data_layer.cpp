#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/FRCNN/frcnn_roi_data_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"

// caffe.proto > LayerParameter > FrcnnRoiDataLayer
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size
// label start from 1

namespace caffe {

namespace Frcnn {

template <typename Dtype>
 FrcnnRoiDataLayer<Dtype>::~FrcnnRoiDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  // roi_data_file format
  // repeated:
  //   # image_index
  //   img_path (rel path)
  //   num_roi
  //   label x1 y1 x2 y2
  
  const int Config_Size = this->layer_param_.window_data_param().config_size();
  CHECK_EQ( Config_Size , 2 ) << "Need Two Config Files, First is Default Param, Second is Override Param";
  std::string default_config_file = this->layer_param_.window_data_param().config( 0 );
  std::string override_config_file = this->layer_param_.window_data_param().config( 1 );
  FrcnnParam::load_param(override_config_file, default_config_file);
  FrcnnParam::print_param();
  cache_images_ = this->layer_param_.window_data_param().cache_images();

  LOG(INFO) << "FrcnnRoiDataLayer :" ;
  LOG(INFO) << "  source file :"
            << this->layer_param_.window_data_param().source() ; 
  LOG(INFO) << "  cache_images: "
            << ( cache_images_ ? "true" : "false" ) ; 
  LOG(INFO) << "  root_folder: "
            << this->layer_param_.window_data_param().root_folder() ;
  LOG(INFO) << "  Default Config File: "
            << default_config_file ;
  LOG(INFO) << "  Override Config File: "
            << override_config_file; 

  const std::string root_folder =
      this->layer_param_.window_data_param().root_folder();

  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open roi_data file "
                       << this->layer_param_.window_data_param().source()
                       << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));
  roi_database_.clear();

  string hashtag;
  int image_index;

  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Roi file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    image_path = root_folder + image_path;

    image_database_.push_back(image_path);
    lines_.push_back(image_index);
    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    // read each box
    int num_roi;
    infile >> num_roi;
    vector<vector<float> > rois;
    for (int i = 0; i < num_roi; ++i) {
      int label, x1, y1, x2, y2;
      infile >> label >> x1 >> y1 >> x2 >> y2;

/////// CHECK LABEL
      vector<float> roi(FrcnnRoiDataLayer::NUM);
      roi[FrcnnRoiDataLayer::LABEL] = label;
      roi[FrcnnRoiDataLayer::X1] = x1;
      roi[FrcnnRoiDataLayer::Y1] = y1;
      roi[FrcnnRoiDataLayer::X2] = x2;
      roi[FrcnnRoiDataLayer::Y2] = y2;

      rois.push_back(roi);
      label_hist.insert(std::make_pair(label, 0));
      label_hist[label]++;
    }
    roi_database_.push_back(rois);

    if (image_index % 1000 == 0) {
      LOG(INFO) << "num: " << image_index << " " << image_path << " "
                << "rois to process: " << num_roi;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "number of images: " << image_index + 1;
  lines_id_ = 0;

  for (map<int, int>::iterator it = label_hist.begin(); it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  // image
  vector<float> scales = FrcnnParam::scales;
  max_short_ = *max_element(scales.begin(), scales.end());
  max_long_ = FrcnnParam::max_size;
  const int batch_size = 1;

  // data mean
  for (int i = 0; i < 3; i++) {
    mean_values_[i] = FrcnnParam::pixel_means[i];
  }

  // data image Input ..
  CHECK_GT(max_short_, 0);
  CHECK_GT(max_long_, 0);

  top[0]->Reshape(batch_size, 3, max_short_, max_long_);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    this->prefetch_[i].data_.Reshape(batch_size, 3, max_short_, max_long_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();

  // im_info: height width scale_factor
  top[1]->Reshape(1, 3, 1, 1);
  // gt_boxes: label x1 y1 x2 y2
  top[2]->Reshape(batch_size, 5, 1, 1);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(batch_size + 1, 5, 1, 1);
  }

  LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = FrcnnParam::rng_seed;
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleImages();
} 

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::ShuffleImages() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
unsigned int FrcnnRoiDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t *prefetch_rng =
      static_cast<caffe::rng_t *>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// This function is called on prefetch thread
template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
  // At each iteration, Give Batch images and
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  const vector<float> scales = FrcnnParam::scales;
  const bool mirror = FrcnnParam::use_flipped;
  const int batch_size = 1;

  timer.Start();
  CHECK_EQ(roi_database_.size(), image_database_.size())
      << "image and roi size abnormal";

  // Select id for batch -> <0 if fliped
  CHECK_LT(lines_id_, lines_.size()) << "select error line id";
  int index = lines_[lines_id_];
  bool do_mirror = mirror && PrefetchRand() % 2;
  float max_short = scales[PrefetchRand() % scales.size()];

  DLOG(ERROR) << "FrcnnRoiDataLayer load batch: " << image_database_[index];
  // label format:
  // labels x1 y1 x2 y2
  // special for frcnn , this first channel is -1 , width , height ,
  // width_with_pad , height_with_pad
  const int channels = roi_database_[index].size() + 1;
  batch->label_.Reshape(channels, 5, 1, 1);
  Dtype *top_label = batch->label_.mutable_cpu_data();
  read_time += timer.MicroSeconds();

  // Prepare Image and labels;
  timer.Start();
  cv::Mat cv_img;
  if (this->cache_images_) {
    pair<std::string, Datum> image_cached = image_database_cache_[index];
    cv_img = DecodeDatumToCVMat(image_cached.second, true);
  } else {
    cv_img = cv::imread(image_database_[index], CV_LOAD_IMAGE_COLOR);
    if (!cv_img.data) {
      LOG(ERROR) << "Could not open or find file " << image_database_[index];
      return;
    }
  }
  cv::Mat src;
  cv_img.convertTo(src, CV_32FC3);
  if (do_mirror) {
    cv::flip(src, src, 1); // Flip
  }
  CHECK(src.isContinuous()) << "Warning : cv::Mat src is not Continuous !";
  CHECK(src.depth() == CV_32F) << "Image data type must be float 32 type";
  read_time += timer.MicroSeconds();

  timer.Start();
  // Convert by : sub means and resize
  // Image sub means
  for (int r = 0; r < src.rows; r++) {
    for (int c = 0; c < src.cols; c++) {
      int offset = (r * src.cols + c) * 3;
      reinterpret_cast<float *>(src.data)[offset + 0] -= this->mean_values_[0]; // B
      reinterpret_cast<float *>(src.data)[offset + 1] -= this->mean_values_[1]; // G
      reinterpret_cast<float *>(src.data)[offset + 2] -= this->mean_values_[2]; // R
    }
  }
  float im_scale = Frcnn::get_scale_factor(src.cols, src.rows, max_short, max_long_);
  cv::resize(src, src, cv::Size(), im_scale, im_scale);

  // resize data
  batch->data_.Reshape(batch_size, 3, src.rows, src.cols);
  Dtype *top_data = batch->data_.mutable_cpu_data();

  for (int r = 0; r < src.rows; r++) {
    for (int c = 0; c < src.cols; c++) {
      int cv_offset = (r * src.cols + c) * 3;
      int blob_shift = r * src.cols + c;
      top_data[0 * src.rows * src.cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 0];
      top_data[1 * src.rows * src.cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 1];
      top_data[2 * src.rows * src.cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 2];
    }
  }

  top_label[0] = src.rows; // height
  top_label[1] = src.cols; // width
  top_label[2] = im_scale; // im_scale: used to filter min size
  top_label[3] = 0;
  top_label[4] = 0;

  const vector<vector<float> > rois = roi_database_[index];
  for (int i = 1; i < channels; i++) {
    CHECK_EQ(rois[i-1].size(), FrcnnRoiDataLayer::NUM);
    top_label[5 * i + 0] = rois[i-1][FrcnnRoiDataLayer::X1] * im_scale; // x1
    top_label[5 * i + 1] = rois[i-1][FrcnnRoiDataLayer::Y1] * im_scale; // y1
    top_label[5 * i + 2] = rois[i-1][FrcnnRoiDataLayer::X2] * im_scale; // x2
    top_label[5 * i + 3] = rois[i-1][FrcnnRoiDataLayer::Y2] * im_scale; // y2
    if (do_mirror) {
      top_label[5 * i + 0] = src.cols - top_label[5 * i + 0];
      top_label[5 * i + 2] = src.cols - top_label[5 * i + 2];
      std::swap(top_label[5 * i + 0], top_label[5 * i + 2]);
    }
    top_label[5 * i + 4] = rois[i-1][FrcnnRoiDataLayer::LABEL];         // label
  }

  trans_time += timer.MicroSeconds();

  lines_id_++;
  if (lines_id_ >= lines_.size()) {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = 0;
    ShuffleImages();
  }

  batch_timer.Stop();
  DLOG(INFO) << "Image Information: " << "height " << top_label[0] << " width "
             << top_label[1] << " scale " << top_label[2];
  DLOG(INFO) << "Ground Truth Boxes: " << channels-1 << " boxes";
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(3, batch->label_.cpu_data(), top[1]->mutable_cpu_data());
    // Reshape to loaded labels.
    vector<int> label_shape(batch->label_.shape());
    label_shape[0] = label_shape[0] - 1;
    top[2]->Reshape(label_shape);
    // Copy the labels.
    caffe_copy(batch->label_.count() - 5, batch->label_.cpu_data() + 5,
               top[2]->mutable_cpu_data());
  }
  this->prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(FrcnnRoiDataLayer, Forward);
#endif

INSTANTIATE_CLASS(FrcnnRoiDataLayer);
REGISTER_LAYER_CLASS(FrcnnRoiData);

} // namespace Frcnn

} // namespace caffe
