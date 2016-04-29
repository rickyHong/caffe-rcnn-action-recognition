#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "api/FRCNN/rpn_api.hpp"

DEFINE_string(gpu, "", 
    "Optional; run in GPU mode on the given device ID, Empty is CPU");
DEFINE_string(model, "", 
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "", 
    "Trained Model By Faster RCNN End-to-End Pipeline.");
DEFINE_string(default_c, "", 
    "Default config file path.");
DEFINE_string(override_c, "", 
    "Override config file path.");
DEFINE_string(image_dir, "",
    "Optional;Test images Dir."); 
DEFINE_string(out_dir, "",
    "Optional;Output images Dir."); 

int main(int argc, char** argv){
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: demo_frcnn_api <args>\n\n"
      "args:\n"
      "  --gpu          7       use 7-th gpu device, default is cpu model\n"
      "  --model        file    protocol buffer text file\n"
      "  --weights      file    Trained Model\n"
      "  --default_c    file    Default Config File\n"
      "  --override_c   file    Override Config File");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  CHECK( FLAGS_gpu.size() == 0 || FLAGS_gpu.size() == 1 ) << "Can only support one gpu or none";
  int gpu_id = -1;
  if( FLAGS_gpu.size() > 0 )
    gpu_id = boost::lexical_cast<int>(FLAGS_gpu);

  std::string proto_file             = FLAGS_model.c_str();
  std::string model_file             = FLAGS_weights.c_str();
  std::string default_config_file    = FLAGS_default_c.c_str();
  std::string override_config_file   = FLAGS_override_c.c_str();

  std::string image_dir = FLAGS_image_dir.c_str();
  std::string out_dir = FLAGS_out_dir.c_str();
  std::vector<std::string> images = caffe::Frcnn::get_file_list(image_dir, ".jpg");
  FRCNN_API::Rpn_Det detector(proto_file, model_file, default_config_file, override_config_file, gpu_id); 
  
  std::vector<caffe::Frcnn::BBox<float> > results;
  caffe::Timer time_;
  DLOG(INFO) << "Test Image Dir : " << image_dir << "  , have " << images.size() << " pictures!";
  DLOG(INFO) << "Output Dir Is : " << out_dir;
  for (size_t index = 0; index < images.size(); ++index) {
    cv::Mat image = cv::imread(image_dir+images[index]);
    time_.Start();
    detector.predict(image, results);
    LOG(INFO) << "Predict " << images[index] << " cost " << time_.MilliSeconds() << " ms."; 
    std::vector<caffe::Frcnn::BBox<float> > results_drop_low_confidence;
    for (size_t obj = 0; obj < results.size(); obj++) {
      if (results[obj].confidence >= caffe::Frcnn::FrcnnParam::test_score_thresh) {
        results_drop_low_confidence.push_back( results[obj] );
      }
    }
    results = results_drop_low_confidence;
    LOG(INFO) << "There are " << results.size() << " objects in picture.";
    for (size_t obj = 0; obj < results.size(); obj++) {
        LOG(INFO) << results[obj].to_string();
    }
    
    for (size_t obj = 0; obj < results.size(); obj++) {
      cv::Mat ori ; 
      image.convertTo(ori, CV_32FC3);
      caffe::Frcnn::vis_detections(ori, results[obj], caffe::Frcnn::LoadRpnClass() );
      std::string name = out_dir+images[index];
      char xx[100];
      sprintf(xx, "%s_%03d.jpg", name.c_str(), (int)obj);
      cv::imwrite(std::string(xx), ori);
    }
  }
  return 0;
}
