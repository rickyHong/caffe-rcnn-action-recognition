#include "api/FRCNN_OLD/Faster_Inference.hpp"

using namespace caffe::FRCNN_OLD;

const std::string WINDOWS = "Faster Detection";
int main(int argc , char **argv){
    LOG(ERROR) << "In Main function !";
    CHECK( argc == 3 ) << "[Detection_Config] [GPU_ID]" << " not " << argc;
    std::stringstream ids( argv[2] );
    int GPU_ID ; ids >> GPU_ID ;
    if( GPU_ID < 0 ){ 
        LOG(ERROR) << "SET CPU MODEL";
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }else{     
        LOG(ERROR) << "SET GPU MODEL , ID: " << GPU_ID;
        caffe::Caffe::SetDevice( GPU_ID );  
    }

    FasterDetector detector ;
    detector.Set_Conf( argv[1] );
    

    cv::namedWindow ( WINDOWS );
    cv::VideoCapture capture( 0 ); // open the default camera
    if(!capture.isOpened()){  // check if we succeeded
        LOG(ERROR) << "Can't open capture! ";
        return -1;
    }
    cv::Mat image ; 
    while( capture.read(image) ){
        //LOG(INFO) << index << "  : " << image_path ;
        vector<BBox> ans = detector.Predict( image ); 
        /*
        for(size_t i = 0 ; i < ans.size() ; i ++ ){
            LOG(INFO) << ans[i].Print( true ) ;
        }
        LOG(INFO) << "--------------\n";
        */
        showImage( image , ans );
        imshow( WINDOWS , image );
        if( cv::waitKey(20) == 27 ){
            LOG(ERROR) << "Type ESC , EXIT!";
            break;
        }
    }
    return 0;
}
