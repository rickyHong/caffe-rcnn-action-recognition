#include "api/FRCNN_OLD/Faster_Inference.hpp"

using namespace caffe::FRCNN_OLD;

int main(int argc , char **argv){
    google::InitGoogleLogging(argv[0]);
    LOG(ERROR) << "In Main function !";
    CHECK( argc == 5 ) << "[Inputfile] [Outputfile] [Detection_Config] [GPU_ID] " << " not " << argc;
    std::stringstream ids( argv[4] );
    int GPU_ID ; ids >> GPU_ID ;
    if( GPU_ID < 0 ){ 
        LOG(ERROR) << "SET CPU MODEL";
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }else{     
        LOG(ERROR) << "SET GPU MODEL , ID: " << GPU_ID;
        caffe::Caffe::SetDevice( GPU_ID );  
        caffe::Caffe::set_mode(caffe::Caffe::GPU);     
    }
    FasterDetector detector ;
    detector.Set_Conf( argv[3] );

    const std::string outputfile( argv[2] );
    const std::string image_path( argv[1] );
    LOG(ERROR) << "Image  : " << image_path ;
    vector<BBox> ans = detector.Predict( image_path ); 
    LOG(ERROR) << "Detect " << ans.size() << " faces.";
    for(size_t i = 0 ; i < ans.size() ; i ++ ){
        LOG(ERROR) << ans[i].Print( true ) ;
    }
    cv::Mat image = cv::imread( image_path );
    showImage( image , ans );
    cv::imwrite( outputfile.c_str() , image );

    return 0;
}
