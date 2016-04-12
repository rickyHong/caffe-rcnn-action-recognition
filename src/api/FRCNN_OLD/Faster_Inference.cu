#include "api/FRCNN_OLD/Faster_Inference.hpp"
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

#define DIVUP(m,n)      ((m)/(n)+((m)%(n)>0))
const int threadsPerBlock = (sizeof(unsigned long long) * 8);

namespace caffe{
namespace FRCNN_OLD{

__global__ void BBox_transform_inv(const int n,const int* SORTED_ID,const float* dev_ANCHORS ,float* dev_anchors, const float* RPN_O_Data, const int feat_stride, const int width ,const int height , const int num_of_anchors , const float* dev_scale_k , const float* dev_Bounds) {
    CUDA_KERNEL_LOOP(shift , n) {
        const int index = SORTED_ID[shift];
        const int j = index % width;
        const int k = index / (width*height);
        const int i = (index%(width*height)) / width;
        float *box = dev_anchors + shift*4;

        box[0] = dev_ANCHORS[k*4+0]+j*feat_stride;
        box[1] = dev_ANCHORS[k*4+1]+i*feat_stride;
        box[2] = dev_ANCHORS[k*4+2]+j*feat_stride;
        box[3] = dev_ANCHORS[k*4+3]+i*feat_stride;

        const float det[4] = { RPN_O_Data[ (k*4+0)*height*width + i*width + j ]
            ,RPN_O_Data[ (k*4+1)*height*width + i*width + j ]
                ,RPN_O_Data[ (k*4+2)*height*width + i*width + j ]
                ,RPN_O_Data[ (k*4+3)*height*width + i*width + j ]};

        float src_w = box[2] - box[0] + 1;
        float src_h = box[3] - box[1] + 1;
        float src_ctr_x = box[0] + 0.5*src_w;
        float src_ctr_y = box[1] + 0.5*src_h;
        float pred_ctr_x = det[0] * src_w + src_ctr_x;
        float pred_ctr_y = det[1] * src_h + src_ctr_y;
        float pred_w = exp(det[2]) * src_w;
        float pred_h = exp(det[3]) * src_h;
        box[0] = pred_ctr_x - 0.5*pred_w;
        box[1] = pred_ctr_y - 0.5*pred_h;
        box[2] = pred_ctr_x + 0.5*pred_w;
        box[3] = pred_ctr_y + 0.5*pred_h;

        box[0] = (box[0]-1)*dev_scale_k[0]+1;  box[0]=max(1.,min(box[0],dev_Bounds[0]));
        box[1] = (box[1]-1)*dev_scale_k[1]+1;  box[1]=max(1.,min(box[1],dev_Bounds[1]));
        box[2] = (box[2]-1)*dev_scale_k[2]+1;  box[2]=max(1.,min(box[2],dev_Bounds[2]));
        box[3] = (box[3]-1)*dev_scale_k[3]+1;  box[3]=max(1.,min(box[3],dev_Bounds[3]));

    }
}

__global__ void GET_INDEX(const int n,int *indices){
    CUDA_KERNEL_LOOP(index , n){
        indices[index] = index;
    }
}

__device__ inline float devIoU(float const * const a, float const * const b){
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thres, const float *dev_boxes, unsigned long long *dev_mask){
    const int row_start = blockIdx.y, col_start = blockIdx.x;
    const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock), col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    //if (row_start > col_start) return;

    __shared__ float block_boxes[threadsPerBlock * 4];
    if (threadIdx.x < col_size)
    {
        block_boxes[threadIdx.x * 4 + 0] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
        block_boxes[threadIdx.x * 4 + 1] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
        block_boxes[threadIdx.x * 4 + 2] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
        block_boxes[threadIdx.x * 4 + 3] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
    }
    __syncthreads();

    if (threadIdx.x < row_size)
    {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float *cur_box = dev_boxes + cur_box_idx * 4;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) start = threadIdx.x + 1;
        for (i = start; i < col_size; i++)
        {
            if (devIoU(cur_box, block_boxes + i * 4) > nms_overlap_thres)
            {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

vector<Point4f> FasterDetector::RPN_Post_GPU(float WIDTH,float HEIGHT,float AFT_W,float AFT_H){
    // box_deltas = permute(box_deltas, [3, 2, 1]);

    // box_deltas = reshape(box_deltas, 4, [])';
    // permute from [width, height, channel] to [channel, height, width]

    LOG(ERROR) << "RPN POST START ";
    const int channes = RPN_output[0]->channels();
    CHECK( channes%4 == 0 ) << "RPN_output[0]->channels() should be divided by 4 " << endl;
    CHECK( RPN_output.size() == 2 ) << "RPN_output.size () == 2" << endl;
    const int height = RPN_output[0]->height();
    const int width = RPN_output[0]->width();
    const vector<vector<float> > ANCHORS = conf.anchors;
    const int num_of_anchors = (int)ANCHORS.size();
    const float *RPN_O_Data = RPN_output[0]->gpu_data() ;
    const int SHIFT = ANCHORS.size()*height*width; 
    const float overlap = conf.overlap1;

    LOG(ERROR) << "--RPN POST INIT";
    // initialize indices vector to [0,1,2,..]

    const int size_of_scores = min( conf.pre_topK, SHIFT );
    CHECK( size_of_scores > 0 ) << "pre_topK >0 && output_map_size >0 ";

    float *keys = NULL;
    int *vals = NULL;
    int *indices;
    float *score;
    cudaMalloc((void **)&score, sizeof(float)*SHIFT);
    cudaMalloc((void **)&indices , sizeof(int)*SHIFT);
    cudaMemcpy( score , RPN_output[1]->gpu_data() + SHIFT , sizeof(float)*SHIFT , cudaMemcpyDefault ); 
    GET_INDEX<<< caffe::CAFFE_GET_BLOCKS(SHIFT), caffe::CAFFE_CUDA_NUM_THREADS>>>(SHIFT,indices);
    cudaDeviceSynchronize();
    //cudaMemcpy( indices , INDICES , sizeof(int)*SHIFT , cudaMemcpyDefault );
    cudaMalloc((void **)&keys, sizeof(float)*SHIFT);
    cudaMalloc((void **)&vals, sizeof(int)*SHIFT);

    cub::DoubleBuffer<float> d_keys(score , keys);
    cub::DoubleBuffer<int> d_values(indices , vals);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, SHIFT);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, SHIFT);
    cudaDeviceSynchronize() ;

    LOG(ERROR) << "--RPN POST SORT DONE : " << SHIFT;


    float *host_ANCHORS = new float[ size_of_scores * 4 ];
    for(int i=0;i<num_of_anchors;i++){
        for(int j=0;j<4;j++){
            host_ANCHORS[i*4+j] = ANCHORS[i][j];
        }
    }

    float *dev_ANCHORS; 
    cudaMalloc((void**) &dev_ANCHORS, sizeof(float)*num_of_anchors*4);
    cudaMemcpy(dev_ANCHORS, host_ANCHORS, sizeof(float)*num_of_anchors*4, cudaMemcpyHostToDevice);
    delete []host_ANCHORS;

    float *dev_anchors;
    cudaMalloc((void**) &dev_anchors, sizeof(float)*size_of_scores*4);

    const float scale_k[4] = { (WIDTH-1)/(AFT_W-1) , (HEIGHT-1)/(AFT_H-1) , (WIDTH-1)/(AFT_W-1) , (HEIGHT-1)/(AFT_H-1) };// = Ask();
    const float Bounds[4]  = { WIDTH , HEIGHT , WIDTH , HEIGHT };// = Ask();

    float *dev_scale_k;
    float *dev_Bounds;
    cudaMalloc((void**) &dev_scale_k, sizeof(float)*4);
    cudaMalloc((void**) &dev_Bounds, sizeof(float)*4);
    cudaMemcpy(dev_scale_k, scale_k, sizeof(float)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Bounds, Bounds, sizeof(float)*4, cudaMemcpyHostToDevice);

    LOG(ERROR) << "BBox_transform_inv: " << size_of_scores ;
    BBox_transform_inv<<< caffe::CAFFE_GET_BLOCKS(size_of_scores), caffe::CAFFE_CUDA_NUM_THREADS>>>(size_of_scores, vals
            ,dev_ANCHORS, dev_anchors
            , RPN_O_Data , conf.feat_stride
            ,width ,height ,num_of_anchors ,dev_scale_k ,dev_Bounds); 

    cudaFree( indices );
    cudaFree( score );
    cudaFree(vals);
    cudaFree(keys);
    cudaFree(d_temp_storage);
    cudaFree(dev_scale_k);
    cudaFree(dev_Bounds);
    cudaFree(dev_ANCHORS);
    cudaDeviceSynchronize() ;

    LOG(ERROR) << "--RPN POST TRANSFORM";

    const int boxes_num = size_of_scores;
    //cout << boxes_num << " Box!" << endl;
    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
    unsigned long long *mask_dev = NULL;
    cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long));
    dim3 blocks(DIVUP(boxes_num, threadsPerBlock), DIVUP(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);
    nms_kernel << <blocks, threads >> >(boxes_num, overlap, dev_anchors, mask_dev);

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

    //std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    unsigned long long  *mask_host = new unsigned long long[boxes_num * col_blocks];
    cudaMemcpy(mask_host, mask_dev, sizeof(unsigned long long) * boxes_num * col_blocks, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    //std::vector<int> keep;
    //keep.reserve(boxes_num);
    int pre_index = 0 ;
    vector<Point4f> boxes;
    for (int i = 0; i < boxes_num; i++){
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))){
            //keep.push_back(i + 1);  // to matlab's index

            cudaMemcpy( Cur_Box_GPU_ + 4*pre_index , dev_anchors + 4*i ,sizeof(float)*4, cudaMemcpyDefault );
            Point4f xx ;
            cudaMemcpy( xx.x , dev_anchors + 4*i , sizeof(float)*4 , cudaMemcpyDeviceToHost );
            boxes.push_back( xx );
            pre_index++;
            if( pre_index >= conf.aft_topK ){
                break;
            }
            unsigned long long *p = mask_host + i * col_blocks ;
            for (int j = nblock; j < col_blocks; j++)
            {
                remv[j] |= p[j];
            }
        }
    }
    delete [] mask_host ;
    cudaFree( dev_anchors );
    cudaFree( mask_dev );

    LOG(ERROR) << "--RPN POST NMS DONE : " << boxes.size();
    // NMS 
    return boxes ;
}

__global__ void Copy_Box_to_Caffe(const int n, const float* X, float* Y, const float scale_factor_) {
    CUDA_KERNEL_LOOP(index, n) {
        Y[index*5+0] = 0;
        Y[index*5+1] = X[index*4+0] * scale_factor_;
        Y[index*5+2] = X[index*4+1] * scale_factor_;
        Y[index*5+3] = X[index*4+2] * scale_factor_;
        Y[index*5+4] = X[index*4+3] * scale_factor_;
    }
}

__global__ void Caffe_Copy_to_Temp_Scores(const int n, float* Y,const float* X, const int sco_num) {
    CUDA_KERNEL_LOOP(index, n) {
        Y[index] = X[index*sco_num+1];
    }
}

std::vector<BBox> FasterDetector::Detection_GPU(vector<Point4f> &boxes,const float scale_factor_,const float WIDTH,const float HEIGHT){

    const int box_num = boxes.size();
    LOG(ERROR) << "DETECTION START : " << box_num ;
    shared_ptr<Blob<float> > Conv5 = RPN_caffe_net->blob_by_name(conf.last_shared_output_blob_name);
    const vector<Blob<float>*>& input_blobs = DET_caffe_net->input_blobs();
    input_blobs[0]->Reshape( Conv5->num() , Conv5->channels() , Conv5->height() , Conv5->width() );
    input_blobs[1]->Reshape(box_num , 5 , 1 , 1);

    Copy_Box_to_Caffe<<< caffe::CAFFE_GET_BLOCKS(box_num), caffe::CAFFE_CUDA_NUM_THREADS>>>( 
            box_num, Cur_Box_GPU_, input_blobs[1]->mutable_gpu_data(), scale_factor_ );

    cudaMemcpy( input_blobs[0]->mutable_gpu_data() , Conv5->gpu_data(), sizeof(float) * input_blobs[0]->count(), cudaMemcpyDefault);

    LOG(ERROR) << "--DET COPY CONV5";
    std::vector<Blob<float>*> DET_output = DET_caffe_net->ForwardPrefilled();
    cudaDeviceSynchronize();
    LOG(ERROR) << "--DET COPY BOXES";

    //const int box_num = DET_output[0]->num();
    const int cls_num = DET_output[0]->channels();
    const int sco_num = DET_output[1]->channels();
    std::vector<BBox> RET;
    CHECK( sco_num > 1 );

    for(int cls = 1 ; cls < sco_num ; cls++ ){
        vector<BBox> xxx;
        for(int i = 0 ; i < box_num ; i++ ){
            //for(int j=1; j < sco_num; j++){
            float scores = DET_output[1]->cpu_data()[i*sco_num+cls];
            Point4f det(
                    DET_output[0]->cpu_data()[ i*cls_num + cls*4+0 ]
                    ,DET_output[0]->cpu_data()[ i*cls_num + cls*4+1 ]
                    ,DET_output[0]->cpu_data()[ i*cls_num + cls*4+2 ]
                    ,DET_output[0]->cpu_data()[ i*cls_num + cls*4+3 ]
                    );
            Point4f cur = bbox_transform_inv( boxes[i], det );
            cur.x[0] = std::max( 1.0f , cur.x[0] );
            cur.x[1] = std::max( 1.0f , cur.x[1] );
            cur.x[2] = std::min( WIDTH , cur.x[2] );
            cur.x[3] = std::min( HEIGHT , cur.x[3] );
            xxx.push_back(BBox(cur , cls , scores));
        }

        sort( xxx.begin() , xxx.end() );
        vector<bool> ok(box_num, true);

        // Apply NMS
        for(int i = 0 ; i < box_num ; i++ ){

            if(ok[i] == false || xxx[i].confidence < conf.Final_Thresh){
                continue ; // Apply Thresh
            }

            for(int j = i+1 ; j < box_num ; j++ ){
                if( ok[j] && IoU_MAX( xxx[i] , xxx[j] ) > conf.overlap2 ){
                    ok[j] = false;
                }
            }
            RET.push_back( xxx[i] );
        }
    }
    LOG(ERROR) << "DETECTION DONE : " << RET.size();
    return RET;
}

}
}
