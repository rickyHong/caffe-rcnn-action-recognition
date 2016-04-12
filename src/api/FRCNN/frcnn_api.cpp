#include "api/FRCNN/frcnn_api.hpp"

namespace FRCNN_API{

void Detector:: preprocess(const cv::Mat &img_in, const int blob_idx) {
  const vector<Blob<float> *> &input_blobs = net_->input_blobs();
  cv::Mat img_out;
  img_in.convertTo(img_out, CV_32FC1);
  CHECK(img_out.isContinuous()) << "Warning : cv::Mat img_out is not Continuous !";
  input_blobs[blob_idx]->Reshape(1, img_out.channels(), img_out.rows, img_out.cols);
  float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
  for (int i = 0; i < img_out.cols * img_out.rows; i++) {
    blob_data[img_out.cols * img_out.rows * 0 + i] =
        reinterpret_cast<float*>(img_out.data)[i * 3 + 0] - mean_[0]; 
    blob_data[img_out.cols * img_out.rows * 1 + i] =
        reinterpret_cast<float*>(img_out.data)[i * 3 + 1] - mean_[1];
    blob_data[img_out.cols * img_out.rows * 2 + i] =
        reinterpret_cast<float*>(img_out.data)[i * 3 + 2] - mean_[2];
  }
}

void Detector::preprocess(const vector<float> &data, const int blob_idx){
  const vector<Blob<float> *> &input_blobs = net_->input_blobs();
  input_blobs[blob_idx]->Reshape(1, data.size(), 1, 1);
  float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
  std::memcpy(blob_data, &data[0], sizeof(float) * data.size());
}

void Detector::Set_Model(std::string &proto_file, std::string &model_file, std::string default_config, std::string override_config, int gpu_id){
  if (gpu_id >= 0) {
#ifndef CPU_ONLY
    caffe::Caffe::SetDevice(gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
    LOG(FATAL) << "CPU ONLY MODEL, BUT PROVIDE GPU ID";
#endif
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
  net_.reset(new Net<float>(proto_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(model_file);
  FrcnnParam::load_param(override_config, default_config); 
  mean_[0] = FrcnnParam::pixel_means[0];
  mean_[1] = FrcnnParam::pixel_means[1];
  mean_[2] = FrcnnParam::pixel_means[2];
  caffe::Frcnn::FrcnnParam::print_param();
  LOG(INFO) << "SET MODEL DONE";
}

vector<boost::shared_ptr<Blob<float> > > Detector::predict(const vector<std::string> blob_names) {
  LOG(ERROR) << "FORWARD BEGIN";
  net_->ForwardPrefilled();
  vector<boost::shared_ptr<Blob<float> > > output;
  for (int i = 0; i < blob_names.size(); ++i) {
    output.push_back(net_->blob_by_name(blob_names[i]));
  }
  LOG(ERROR) << "FORWARD END";
  return output;
}

void Detector::predict(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results){

  CHECK(FrcnnParam::test_scales.size() == 1) << "Only single-image batch implemented";

  float scale_factor = caffe::Frcnn::get_scale_factor(img_in.cols, img_in.rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);

  cv::Mat img;
  const float height = img_in.rows;
  const float width = img_in.cols;
  LOG(ERROR) << "height: " << height << " width: " << width;

  cv::resize(img_in, img, cv::Size(), scale_factor, scale_factor);

  std::vector<float> im_info(3);
  im_info[0] = img.rows;
  im_info[1] = img.cols;
  im_info[2] = scale_factor;

  this->preprocess(img, 0);
  this->preprocess(im_info, 1);

  vector<std::string> blob_names(3);
  blob_names[0] = "rois";
  blob_names[1] = "cls_prob";
  blob_names[2] = "bbox_pred";

  vector<boost::shared_ptr<Blob<float> > > output = this->predict(blob_names);
  boost::shared_ptr<Blob<float> > rois(output[0]);
  boost::shared_ptr<Blob<float> > cls_prob(output[1]);
  boost::shared_ptr<Blob<float> > bbox_pred(output[2]);

  const int box_num = bbox_pred->num();
  const int cls_num = cls_prob->channels();
  CHECK_GT(cls_num , 0);
  results.clear();
  
  for (int cls = 1; cls < cls_num; cls++) { 
    vector<BBox<float> > bbox;
    for (int i = 0; i < box_num; i++) { 
      float score = cls_prob->cpu_data()[i * cls_num + cls];

      Point4f<float> roi(rois->cpu_data()[(i * 5) + 1]/scale_factor,
                     rois->cpu_data()[(i * 5) + 2]/scale_factor,
                     rois->cpu_data()[(i * 5) + 3]/scale_factor,
                     rois->cpu_data()[(i * 5) + 4]/scale_factor);

      Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 0],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 1],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 2],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 3]);

      Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
      box[0] = std::max(1.0f, box[0]);
      box[1] = std::max(1.0f, box[1]);
      box[2] = std::min(width, box[2]);
      box[3] = std::min(height, box[3]);

      // BBox tmp(box, score, cls);
      // LOG(ERROR) << "cls: " << tmp.id << " score: " << tmp.confidence;
      // LOG(ERROR) << "roi: " << roi.to_string();
      bbox.push_back(BBox<float>(box, score, cls));
    }
    sort(bbox.begin(), bbox.end());
    vector<bool> select(box_num, true);
    // Apply NMS
    for (int i = 0; i < box_num; i++)
      if (select[i]) {
        if (bbox[i].confidence < FrcnnParam::test_score_thresh)
          break;
        for (int j = i + 1; j < box_num; j++)
          if (select[j]) {
            if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
              select[j] = false;
            }
          }
        results.push_back(bbox[i]);
      }
  }

}

} // FRCNN_API
