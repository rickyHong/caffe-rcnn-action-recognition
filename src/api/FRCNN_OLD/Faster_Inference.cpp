#include "api/FRCNN_OLD/Faster_Inference.hpp"

namespace caffe{
namespace FRCNN_OLD{

void Config::Display(){
    cout << "Scales: " << SCALES << " , MAX_SIZE: " << MAX_SIZE << endl;
    cout << "RPN Test Prototxt : " << RPN_proto_name << endl;
    cout << "RPN Caffe Model : " << RPN_model_name << endl;
    cout << "Fast Test Prototxt : " << DET_proto_name << endl;
    cout << "Fast Caffe Model : " << DET_model_name << endl;
    cout << "Per Nms:" << pre_topK << ",Aft Nms:" << aft_topK << endl;
    cout << "feat_stride: " << feat_stride << endl;
    cout << "Last Shared: " << last_shared_output_blob_name << endl;
    cout << "Means Pixs:  " << pixel_means_[0] << "  ,  "  << pixel_means_[1] << "  ,  " << pixel_means_[2] << endl;
    cout << "Overlap   :  " << overlap1 << " , " << overlap2 << endl;
}

Config::Config(const string confpath){
    std::ifstream CONFIG( confpath.c_str() );
    CONFIG >> SCALES >> MAX_SIZE ;
    string Model_Path ;//= "/home/sensetime/Documents/RPN4Face_Temp/RPN/RPN_Post/"
    CONFIG >> Model_Path;
    CONFIG >> RPN_proto_name >> DET_proto_name >> RPN_model_name >> DET_model_name;
    RPN_proto_name = Model_Path + RPN_proto_name;
    DET_proto_name = Model_Path + DET_proto_name;
    RPN_model_name = Model_Path + RPN_model_name;
    DET_model_name = Model_Path + DET_model_name;
    feat_stride = 16;
    overlap1 = 0.7;
    overlap2 = 0.3;
    pre_topK = 100;
    aft_topK = 10;
    test_min_box_size = 16;
    last_shared_output_blob_name = "conv5";
    CONFIG >> last_shared_output_blob_name;
    // Load Anchors
    int num_of_anchors;
    CONFIG >> num_of_anchors;
    anchors.resize( num_of_anchors );
    for(int i=0;i<num_of_anchors;i++){
        anchors[i].resize(4);
        CONFIG >> anchors[i][0] >> anchors[i][1] >> anchors[i][2] >> anchors[i][3];
    }
    //CONFIG >> GPU_id ;
    CONFIG >> pre_topK >> overlap1 >> aft_topK ;
    CONFIG >> overlap2;
    Final_Thresh = 0.5;
    CONFIG >> Final_Thresh;
    CONFIG >> pixel_means_[0] >> pixel_means_[1] >> pixel_means_[2];
    Display();
}

vector<Point4f> FasterDetector::RPN_Post(float WIDTH,float HEIGHT,float AFT_W,float AFT_H){
    const int channes = RPN_output[0]->channels();
    CHECK( channes%4 == 0 ) << "RPN_output[0]->channels() should be divided by 4 " << endl;
    const int height = RPN_output[0]->height();
    const int width = RPN_output[0]->width();
    const float *RPN_O_Data = RPN_output[0]->cpu_data() ;
    const int num_of_anchors = (int)conf.anchors.size();

    typedef pair<float,int> SORT_ID;
    vector<SORT_ID> id;
    std::priority_queue< SORT_ID , vector<SORT_ID>, std::greater<SORT_ID> > que;
    for(int i=0;i<width;i++){
        for(int j=0;j<height;j++){
            for(int k=0;k<num_of_anchors;k++){
                float scores = RPN_output[1]->cpu_data()[ conf.anchors.size()*height*width + k*width*height+ j*width + i ] ;
                const int index = i*height*num_of_anchors+ j * num_of_anchors + k;
                que.push( SORT_ID( scores , index ) );
                while( que.size() > conf.pre_topK ){
                    que.pop();
                }
            }
        }
    }
    while( que.empty() == false ){
        id.push_back( que.top() );
        que.pop();
    }
    reverse( id.begin(), id.end() );

    const int cnt = id.size();
    std::vector<Point4f> anchors( cnt );
    for( size_t index = 0 ; index < id.size() ; index ++ ){
        int pick = id[index].second;
        // if( index < 10 )  LOG(ERROR) << "DEBUG: " << index << "[]" << pick << "{}" << id[index].first;
        int j = pick / (height*num_of_anchors ) ;
        int i = (pick%(height*num_of_anchors)) / num_of_anchors ;
        int k = pick % num_of_anchors ;

        Point4f anchor(conf.anchors[k][0] + j*conf.feat_stride   //shift_x[i][j];
            ,conf.anchors[k][1] + i*conf.feat_stride             //shift_y[i][j];
            ,conf.anchors[k][2] + j*conf.feat_stride             //shift_x[i][j];
            ,conf.anchors[k][3] + i*conf.feat_stride);           //shift_y[i][j];
        Point4f box_deltas(
             RPN_O_Data[ (k*4+0)*height*width + i*width + j ]
            ,RPN_O_Data[ (k*4+1)*height*width + i*width + j ]
            ,RPN_O_Data[ (k*4+2)*height*width + i*width + j ]
            ,RPN_O_Data[ (k*4+3)*height*width + i*width + j ]
            );
        anchors[index] = bbox_transform_inv( anchor , box_deltas );
    }

    // Scale Back prebox =  anchors
    const float scale_k[4] = { (WIDTH-1)/(AFT_W-1) , (HEIGHT-1)/(AFT_H-1) , (WIDTH-1)/(AFT_W-1) , (HEIGHT-1)/(AFT_H-1) };// = Ask();
    const float Bounds[4]  = { WIDTH , HEIGHT , WIDTH , HEIGHT };// = Ask();
    for(int i = 0 ; i < cnt ; i++ ){
        for(int jj = 0 ; jj < 4 ; jj ++){
            anchors[i].x[jj] = (anchors[i][jj]-1)*scale_k[jj]+1;
            anchors[i].x[jj] = std::max( 1.0f , std::min( anchors[i][jj] , Bounds[i%4] ) );
        }
    }
    std::vector<bool> selet(cnt,true);
    // Apply Nms
    std::vector<Point4f> ans;
    for(int i=0; i<cnt && ans.size() < conf.aft_topK ; i++) if( selet[i] ){
        for(int j=i+1;j<cnt;j++)if(selet[j]){
            if( IoU( anchors[i], anchors[j]) > conf.overlap1){
                selet[j] = false;
            }
        }
        ans.push_back( anchors[i] );
    }

    LOG(ERROR) << "RPN POST : " << ans.size() ;
    return ans ;
}

std::vector<BBox> FasterDetector::Detection(vector<Point4f> &boxes,const float scale_factor_,const float WIDTH,const float HEIGHT){

    const shared_ptr<Blob<float> > Conv5 = RPN_caffe_net->blob_by_name(conf.last_shared_output_blob_name);
    const int box_num_ = boxes.size();
    const vector<Blob<float>*>& input_blobs = DET_caffe_net->input_blobs();
    input_blobs[0]->Reshape(Conv5->num() , Conv5->channels() , Conv5->height() , Conv5->width() );
    if( Caffe::mode() == Caffe::CPU )
        memcpy(input_blobs[0]->mutable_cpu_data(), Conv5->cpu_data(), sizeof(float) * input_blobs[0]->count());
    else{
#ifndef CPU_ONLY
        cudaMemcpy( input_blobs[0]->mutable_gpu_data(), Conv5->gpu_data(), sizeof(float) * input_blobs[0]->count() , cudaMemcpyDefault);
#endif
    }
    input_blobs[1]->Reshape(box_num_ , 5 , 1 , 1 );

    float *roi_input_ = input_blobs[1]->mutable_cpu_data() ; 
    for (int i=0; i< box_num_; i++) {
        roi_input_[i*5] = 0;
        roi_input_[i*5+1] = boxes[i][0] * scale_factor_;
        roi_input_[i*5+2] = boxes[i][1] * scale_factor_;
        roi_input_[i*5+3] = boxes[i][2] * scale_factor_;
        roi_input_[i*5+4] = boxes[i][3] * scale_factor_;
    }
    
    const vector<Blob<float>*> & DET_output = DET_caffe_net->ForwardPrefilled();
    const int box_num = DET_output[0]->num();
    const int cls_num = DET_output[0]->channels();
    const int sco_num = DET_output[1]->channels();
    std::vector<BBox> RET;
    CHECK( sco_num > 1 );

    for(int cls = 1 ; cls < sco_num ; cls++){
        vector<BBox> xxx;
        for(int i=0; i < box_num ; i++){
            //for(int j=1; j < sco_num; j++){
            float scores = DET_output[1]->cpu_data()[i*sco_num+cls];
            Point4f det(
                 DET_output[0]->cpu_data()[ i*cls_num + cls*4+0 ]
                ,DET_output[0]->cpu_data()[ i*cls_num + cls*4+1 ]
                ,DET_output[0]->cpu_data()[ i*cls_num + cls*4+2 ]
                ,DET_output[0]->cpu_data()[ i*cls_num + cls*4+3 ]);
            Point4f cur = bbox_transform_inv( boxes[i], det );
            cur.x[0] = std::max( 1.0f , cur.x[0] );
            cur.x[1] = std::max( 1.0f , cur.x[1] );
            cur.x[2] = std::min( WIDTH , cur.x[2] );
            cur.x[3] = std::min( HEIGHT , cur.x[3] );
            xxx.push_back(BBox(cur , cls , scores));
        }
        sort(xxx.begin(),xxx.end());
        vector<bool> ok(box_num,true);
        // Apply NMS
        for(int i=0;i<box_num;i++)if(ok[i]){
            if(xxx[i].confidence < conf.Final_Thresh) break; // Apply Thresh
            for(int j=i+1;j<box_num;j++)if(ok[j]){
                if( IoU_MAX(xxx[i] , xxx[j]) > conf.overlap2 ){
                    ok[j] = false;
                }
            }
            RET.push_back( xxx[i] );
        }
    }

    return RET;
}

vector<BBox> FasterDetector::Predict(const string &image_path){
    cv::Mat imgin = cv::imread(image_path);
    vector<BBox> ans = Predict(imgin);
    return ans;
}

vector<BBox> FasterDetector::Predict(const cv::Mat& imgin ){
    // Get_Image_Input
    LOG(ERROR)<<"COVERT DATA BEGIN ";
    cv::Mat imgout;
    imgin.convertTo(imgout, CV_32FC3);
    LOG(ERROR)<<"COVERT DATA FINISHED";

    float short_len = std::min(imgout.cols, imgout.rows);
    float long_len = std::max(imgout.cols, imgout.rows);

    float WIDHT = imgout.cols;
    float HEIGHT = imgout.rows;

    
    float scale_factor_ = conf.SCALES / short_len;
    if (long_len*scale_factor_ > conf.MAX_SIZE){
        scale_factor_ = conf.MAX_SIZE / long_len;
    }
    /*
    float scale_factor_ = 1.0;
    if( conf.SCALES > short_len )
        scale_factor_ = conf.SCALES / short_len;
    else if( conf.MAX_SIZE < long_len ){
        scale_factor_ = conf.MAX_SIZE / long_len;
    } 

    */
    LOG(ERROR)<<"RESIZE , SUBSTRACT , COPY DATA BEGIN :" << scale_factor_;
    cv::resize(imgout, imgout, cv::Size(), scale_factor_, scale_factor_);
    const vector<Blob<float>*>& input_blobs = RPN_caffe_net->input_blobs();
    CHECK ( input_blobs.size() == 1) << "illegal RPN input layer size!" << input_blobs.size() << endl;
    input_blobs[0]->Reshape( 1 , imgout.channels() , imgout.rows , imgout.cols );
    float *img_input_ = input_blobs[0]->mutable_cpu_data() ;
    for (int i=0; i<imgout.cols*imgout.rows; i++) {
        img_input_[imgout.cols*imgout.rows*0 + i] = ((float*)imgout.data)[i*3+0] - conf.pixel_means_[0];
        img_input_[imgout.cols*imgout.rows*1 + i] = ((float*)imgout.data)[i*3+1] - conf.pixel_means_[1];
        img_input_[imgout.cols*imgout.rows*2 + i] = ((float*)imgout.data)[i*3+2] - conf.pixel_means_[2];
    }
    LOG(ERROR)<<"RESIZE , SUBSTRACT , COPY DATA FINISHED->(w,h):"<<imgout.cols<<","<<imgout.rows;

    LOG(ERROR) << "RPN FORWARD BEGIN";
    RPN_output = RPN_caffe_net->ForwardPrefilled();
    LOG(ERROR) << "RPN FORWARD FINISHED";

    vector<BBox> ans ;
    if( Caffe::mode() == Caffe::CPU ){
        vector<Point4f> proposal = RPN_Post(WIDHT,HEIGHT,imgout.cols,imgout.rows);
        ans = Detection( proposal ,scale_factor_, WIDHT, HEIGHT);
    }else{
#ifndef CPU_ONLY
    // For GPU , sync 
        vector<Point4f> proposal = RPN_Post_GPU(WIDHT,HEIGHT,imgout.cols,imgout.rows);
        ans = Detection_GPU( proposal ,scale_factor_, WIDHT, HEIGHT);
#endif
    }
    return ans;
}

void FasterDetector::Set_Conf(const string conf_path){
    conf = Config(conf_path);

    DET_caffe_net.reset(new Net<float>( conf.DET_proto_name , TEST )); 
    DET_caffe_net->CopyTrainedLayersFrom( conf.DET_model_name );

    RPN_caffe_net.reset(new Net<float>( conf.RPN_proto_name , TEST )); 
    RPN_caffe_net->CopyTrainedLayersFrom( conf.RPN_model_name );

#ifndef CPU_ONLY
    if(Cur_Box_GPU_ != NULL)
        cudaFree(Cur_Box_GPU_);
    cudaMalloc((void**)&Cur_Box_GPU_ , 4 * 5 * conf.aft_topK * sizeof(float));
#endif
    LOG(ERROR) << "SET CONFIG AND MODELS DONE!";
}

}
}
