#include <iostream>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/SpeedUp/math_functions_sparse.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
    FLAGS_alsologtostderr = 1;
    ::google::InitGoogleLogging(argv[0]);
    //caffe::GlobalInit(&argc, &argv);

    Sparse_Matrix sparse;
#ifdef USE_MKL
    sparse.TestConvert();
#endif
    double sparse_ratio;
    do{
        LOG(INFO) << "Input [Sparse Ratio, M, N]  (negative will end the process) : ";
        int M , N ;
        std::cin >> sparse_ratio >> M >> N;
        if( M < 0 || N < 0 ) break;
        sparse.TestMVProduct(M, N, sparse_ratio);
    }while( sparse_ratio>=0 && sparse_ratio <=1 );

    do{
        LOG(INFO) << "Input [Spars Ratio[0], M, K, N]  (negative will end the process) : ";
        LOG(INFO) << "Test MM,  C = alpha * A * B + beta *C ,  A is m-by-k, B is k-by-n, C is m-by-n";
        int M, N, K;
        std::cin >> sparse_ratio >> M >> K >> N;
        if( N < 0 ) break;
        sparse.TestMMProduct(M, K, N, sparse_ratio);
    }while( sparse_ratio>=0 && sparse_ratio <=1 );

/*
    vector<MM_Time> Save;
    int M, N, K;
    M = 1000;
    for ( N = 300; N < 3000; N += 100) { 
        for ( K = 300; K < 3000; K+= 100) {
            MM_Time time_ = sparse.TestMMProduct(M, K, N, 0);
            Save.push_back(time_);
        }
    }
    N = 1000;
    for ( M = 300; M < 3000; M += 100) { 
        for ( K = 300; K < 3000; K+= 100) {
            MM_Time time_ = sparse.TestMMProduct(M, K, N, 0);
            Save.push_back(time_);
        }
    }
    K = 1000;
    for ( N = 300; N < 3000; N += 100) { 
        for ( M = 300; M < 3000; M+= 100) {
            MM_Time time_ = sparse.TestMMProduct(M, K, N, 0);
            Save.push_back(time_);
        }
    }
    for (size_t index = 0; index < Save.size(); index++) {
        LOG(ERROR) << Save[index].show();
    }
*/
    return 0;
}
