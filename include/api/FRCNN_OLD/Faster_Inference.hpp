#ifndef FASTER_INFERENCE_
#define FASTER_INFERENCE_
#include <iostream>
#include <cstdio>
#include <queue>
#include <cmath>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"

namespace caffe{
namespace FRCNN_OLD{

using namespace std;

class Point4f{
public:
    float x[4];
    Point4f(float x1=0,float y1=0,float x2=0,float y2=0){
        x[0] = x1 ; x[1] = y1 ;
        x[2] = x2 ; x[3] = y2 ;
    }
    Point4f(const float y[4]){
        memcpy( x , y , 4 * sizeof(float) );
    }
    Point4f(const Point4f& other){
        memcpy( x , other.x , 4 * sizeof(float) );
        
    }
    class Point4f operator * (const float scale) const{
        return Point4f(x[0]*scale , x[1]*scale , x[2]*scale , x[3]*scale );
    }
    float operator [] (const int id) const {
        return x[id];
    }
    std::string Print()const {
        char s[80];
        sprintf(s,"%.2f %.2f %.2f %.2f",x[0],x[1],x[2],x[3]);
        return std::string(s);
    }
    // x1 y1 x2 y2 
    // w h x_ctr y_ctr
} ;

class BBox : public Point4f{
public:
    int classes ;
    float confidence ;
    BBox(float _x1=0,float _y1=0,float _x2=0,float _y2=0,float scores=0):Point4f(_x1,_y1,_x2,_y2),confidence(scores){}
    BBox(float BOX[5]):Point4f(BOX),confidence(BOX[4]){}
    BBox(Point4f xx,int _cls,float _conf):Point4f(xx),classes(_cls) , confidence(_conf){
    }   
    BBox(float _x1=0,float _y1=0,float _x2=0,float _y2=0,float scores=0,int _cls=0):Point4f(_x1,_y1,_x2,_y2),classes(_cls),confidence(scores){}
    class BBox operator * (const float scale) const{
        return BBox( scale*x[0] , scale*x[1] , scale*x[2] , scale*x[3] , classes , confidence);
    }   
    bool operator < (const class BBox & y)const{
        if( confidence != y.confidence )
            return confidence > y.confidence;
        else return classes < y.classes;
    }   
    BBox& operator=(const BBox& other){
        memcpy( x , other.x , sizeof(x) );
        confidence = other.confidence;
        return *this;
    }   
    std::string Print(const bool addPos ) const{
        char s[90];
        if( addPos ){
            sprintf(s,"Cls:%3d (%.2f):%.2f %.2f %.2f %.2f",classes,confidence,x[0],x[1],x[2],x[3]);
        }else{
            sprintf(s,"Cls:%3d (%.3f)",classes,confidence);
        }   
        return std::string(s);
    }
};

inline float IoU(const Point4f &A,const Point4f &B){  
    const float xx1 = std::max(A[0],B[0]);
    const float yy1 = std::max(A[1],B[1]); 
    const float xx2 = std::min(A[2],B[2]);
    const float yy2 = std::min(A[3],B[3]);
    float inter = std::max(0.0f,xx2-xx1+1)*std::max(0.0f,yy2-yy1+1);  
    float areaA = (A[2]-A[0]+1)*(A[3]-A[1]+1);             
    float areaB = (B[2]-B[0]+1)*(B[3]-B[1]+1);        
    return inter / (areaA + areaB - inter);  
}; 

inline float IoU_MAX(const Point4f &A,const Point4f &B){  
    const float xx1 = std::max(A[0],B[0]);
    const float yy1 = std::max(A[1],B[1]); 
    const float xx2 = std::min(A[2],B[2]);
    const float yy2 = std::min(A[3],B[3]);
    float inter = std::max(0.0f,xx2-xx1+1)*std::max(0.0f,yy2-yy1+1);
    float areaA = (A[2]-A[0]+1)*(A[3]-A[1]+1);         
    float areaB = (B[2]-B[0]+1)*(B[3]-B[1]+1);
    return std::max( inter / areaA , inter / areaB ); 
};

inline void showImage(cv::Mat & frame,const std::vector<BBox>& ans){
    for(size_t i = 0 ; i < ans.size() ; i++){
        cv::rectangle(frame, cv::Point(ans[i][0],ans[i][1]) , cv::Point(ans[i][2],ans[i][3]) , cv::Scalar(255,0,0) );
        const string contxt = ans[i].Print( false );
        cv::putText(frame, contxt.c_str() , cv::Point(ans[i][0],ans[i][1]) , 0 , 0.6 , cv::Scalar(0,255,0) );
    }   

}

class Config{
public:
    float SCALES;
    float MAX_SIZE;
    float pixel_means_[3];
    float overlap1;
    float overlap2;
    int pre_topK,aft_topK;
    std::string RPN_proto_name,DET_proto_name;
    std::string RPN_model_name,DET_model_name;
    float test_min_box_size;
    int feat_stride;
    std::vector<vector<float> > anchors;
    string last_shared_output_blob_name;
    float Final_Thresh;
    Config(){}
    ~Config(){}
    void Display();
    Config(const string confpath);
};

inline Point4f bbox_transform_inv(const Point4f box, const Point4f det){
    float src_w = box[2] - box[0] + 1;
    float src_h = box[3] - box[1] + 1;
    float src_ctr_x = box[0] + 0.5*src_w;//box[0] + 0.5*src_w;
    float src_ctr_y = box[1] + 0.5*src_h;//box[1] + 0.5*src_h;
    float pred_ctr_x = det[0] * src_w + src_ctr_x;
    float pred_ctr_y = det[1] * src_h + src_ctr_y;
    float pred_w = exp(det[2]) * src_w;
    float pred_h = exp(det[3]) * src_h;
    return Point4f(pred_ctr_x - 0.5*pred_w , pred_ctr_y - 0.5*pred_h , pred_ctr_x + 0.5*pred_w , pred_ctr_y + 0.5*pred_h);
}

inline vector<Point4f> bbox_transform_inv(const Point4f box, const vector<Point4f> deltas){
    vector<Point4f> ans;
    for(size_t index = 0 ; index < deltas.size() ; index ++ ){
        ans.push_back( bbox_transform_inv(box , deltas[index]) );
    }
    return ans;
}

inline Point4f bbox_transform(const Point4f ex_rois,const Point4f gt_rois){
    float ex_widths = ex_rois[2] - ex_rois[0] + 1;
    float ex_heights = ex_rois[3] - ex_rois[1] + 1;
    float ex_ctr_x = ex_rois[0] + 0.5 * ex_widths ;
    float ex_ctr_y = ex_rois[1] + 0.5 * ex_heights ; 
    float gt_widths = gt_rois[2] - gt_rois[0] + 1;
    float gt_heights = gt_rois[3] - gt_rois[1] + 1;
    float gt_ctr_x = gt_rois[0] + 0.5 * gt_widths ;
    float gt_ctr_y = gt_rois[1] + 0.5 * gt_heights ;
    float targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths;
    float targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights;
    float targets_dw = log( gt_widths / ex_widths );
    float targets_dh = log( gt_heights / ex_heights );
    return Point4f(targets_dx,targets_dy,targets_dw,targets_dh);
}

class FasterDetector{
public:
    Config conf;
    FasterDetector (){
#ifndef CPU_ONLY
        Cur_Box_GPU_ = NULL;
        LOG(ERROR) << "FasterDetector Generate! ";
#endif
    }
    ~FasterDetector(){
    }
    void Set_Conf(const string conf_path) ;

    std::vector<BBox> Predict(const string&  image_path);
    std::vector<BBox> Predict(const cv::Mat&  imgin);

private:
    shared_ptr<Net<float> > RPN_caffe_net,DET_caffe_net;
    vector<Blob<float>*> RPN_output,DET_output;
    void set_model(const int);
    vector<BBox> Detection(vector<Point4f> &boxes,const float scale_factor_ ,const float WIDTH ,const float HEIGHT );
    vector<Point4f> RPN_Post(float WIDTH,float HEIGHT,float AFT_W,float AFT_H);

#ifndef CPU_ONLY
    vector<BBox> Detection_GPU(vector<Point4f> &boxes ,float scale_factor_,const float WIDTH,const float HEIGHT);
    vector<Point4f> RPN_Post_GPU(float WIDTH,float HEIGHT,float AFT_W,float AFT_H);
    float * Cur_Box_GPU_;
#endif
    
};

}
}

#endif 
