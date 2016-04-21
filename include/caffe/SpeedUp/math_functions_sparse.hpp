#ifndef CAFFE_UTIL_MATH_FUNCTIONS_SPARSE_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_SPARSE_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <fstream>
#include <sstream>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#ifdef USE_MKL
#include <mkl.h>
#endif  // USE_MKL

namespace caffe {
/*
The Intel MKL compressed sparse row (CSR) format is specified by four arrays: the values, columns, pointerB, and pointerE. The following table describes the arrays in terms of the values, row, and column positions of the non-zero elements in a sparse matrix A.
values    : A real or complex array that contains the non-zero elements of A. Values of the non-zero elements of A are mapped into the values array using the row-major storage mapping described above.
columns   : Element i of the integer array columns is the number of the column in A that contains the i-th value in the values array.
pointerB  : Element j of this integer array gives the index of the element in the values array that is first non-zero element in a row j of A. Note that this index is equal to pointerB(j) - pointerB(1)+1 .
pointerE  : An integer array that contains row indices, such that pointerE(j)-pointerB(1) is the index of the element in the values array that is last non-zero element in a row j of A.

Available in three or four vector formats
Four vector format is also called NIST Blas format.
Three vector format if called CSR3

Three Array Variation of CSR Format:
values    :
columns   :
rowIndex  : 
*/ 

class MM_Time {
public:

  MM_Time(int _m, int _n, int _k, double _time) {
    m = _m; n = _n; k = _k; ms_time = _time;
  } 
  inline std::string show() {
    std::ostringstream out;
    out << m << "    " << k << "    " << n << "    " << ms_time/1000.; // convert to ms
    return std::string(out.str());
  }
private:
  int m,n,k;
  double ms_time;
};

class Sparse_Matrix {
    // CSR3 format , All for zero-based indexing
public: 
    //Dtype *values;
    //int *colums;
    //int *rowIndex;
    void TestConvert();
    void TestMVProduct(int m, int n, const double zero_ratio = 0.3);
    MM_Time TestMMProduct(const int m, const int k, const int n , const double zero_ratio = 0.3);
    int Convert(float *A, const int, const int, float *acsr, int *columns, int *rowIndex);
    //void DisplayMatrix(const float *A, const int m, const int n);
};


}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_SPARSE_H_
