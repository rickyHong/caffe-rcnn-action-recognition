#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/SpeedUp/math_functions_sparse.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

    template <typename Dtype> 
    void DisplayMatrix(const Dtype *A, const int m, const int n){
        if( m*n >= 100 ){
            LOG(INFO) << "Too Many to display.";
            return;
        }
        for (int index = 0; index < m; ++index){
            std::ostringstream buffer;
            for(int jj = 0; jj < n; ++jj){
                int x = A[index*n+jj] * 100;
                buffer << std::setfill(' ') << std::setw(6) << x/100.f;            
            }
            LOG(INFO) << buffer.str(); 
        }
    }
    template void DisplayMatrix<float>(const float *A, const int m, const int n);
    template void DisplayMatrix<int>(const int *A, const int m, const int n);
    template void DisplayMatrix<double>(const double *A, const int m, const int n);

    int GetRandomMatrix(float *A, const int m, const int n, const int seed = 0,const double zero_ratio = 0){
        caffe::rng_t RD(seed); 
        int count_nozeros = m * n;
        for (int index = 0; index < m*n; ++index){
            A[index] = (RD()%5000+1) / 1000.;
            if( RD()%1000 < zero_ratio*1000 ) A[index] = 0;
            if( A[index] == 0 ) count_nozeros --;
        }
        return count_nozeros;
    }

    int Sparse_Matrix::Convert(float *A, const int m, const int n, float *acsr, int *columns, int *rowIndex ){
#ifdef USE_MKL
        int job[] = { 0 , //If job(1)=0, the rectangular matrix A is converted to the CSR format;
            //if job(1)=1, the rectangular matrix A is restored from the CSR format.
            0 , //If job(2)=0, zero-based indexing for the rectangular matrix A is used;
            //if job(2)=1, one-based indexing for the rectangular matrix A is used.
            0 , //If job(3)=0, zero-based indexing for the matrix in CSR format is used;
            //if job(3)=1, one-based indexing for the matrix in CSR format is used.
            2 , //If job(4)=0, adns is a lower triangular part of matrix A;  
            //If job(4)=1, adns is an upper triangular part of matrix A;
            m*n, //If job(4)=2, adns is a whole matrix A.
            //job(5)=nzmax - maximum number of the non-zero elements allowed if job(1)=0.
            1 };//job(6) - job indicator for conversion to CSR format.
        //If job(6)=0, only array ia is generated for the output storage.
        //If job(6)>0, arrays acsr, ia, ja are generated for the output storage.
        int lda = n, info = -1;
        mkl_sdnscsr(job, &m, &n, A, &lda, acsr, columns, rowIndex, &info);
        return info;
#else 
        LOG(FATAL) << "This func requires MKL; compile with BLAS: mkl.";
        return -1;
#endif
    } 

#ifdef USE_MKL
    void Sparse_Matrix::TestConvert(){
        float A[] = {  1, -1,  0, -3,  0,
            -2,  5,  0,  0,  0,
            0,  0,  4,  6,  4,
            -4,  0,  2,  7,  0,
            0,  8,  0,  0, -5}; // 5 * 5 matrix
        const float values[]  =  {1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5};
        const int columns[]     =  {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
        const int rowIndex[]    =  {0, 3, 5, 8, 11, 13};
        //void mkl_sdnscsr(int *job, int *m, int *n, float *adns, int *lda, float *acsr, int *ja, int *ia, int *info);
        const int m = 5;  // INTEGER. Number of rows of the matrix A.
        const int n = 5;  // INTEGER. Number of columns of the matrix A.
        //float adns[20] ;  //(input/output) Array containing non-zero elements of the matrix A.
        //int lda = m;  //(input/output)INTEGER. Specifies the first dimension of adns as declared in the calling (sub)program, must be at least max(1, m).
        float acsr[20] ;  // Array containing non-zero elements of the matrix A. Its length is equal to the number of non-zero elements in the matrix A. Refer to values array description in Sparse Matrix Storage Formats for more details.
        int ja[20] ; //(input/output)INTEGER. Array containing the column indices for each non-zero element of the matrix A.
        //Its length is equal to the length of the array acsr. Refer to columns array description in Sparse Matrix Storage Formats for more details.
        int ia[20] ; //(input/output)INTEGER. Array of length m + 1, containing indices of elements in the array acsr, such that ia(I) is the index in the array acsr of the first non-zero element from the row I. The value of the last element ia(m + 1) is equal to the number of non-zeros plus one. Refer to rowIndex array description in Sparse Matrix Storage Formats for more details.
        int info = -1; //INTEGER. Integer info indicator only for restoring the matrix A from the CSR format.
        //If info=0, the execution is successful.
        //If info=i, the routine is interrupted processing the i-th row because there is no space in the arrays adns and ja according to the value nzmax.

        double Average_time = 0;
        const int LOOP = 50, gap = 10;
        LOG(INFO) << "Matrix A is :";
        DisplayMatrix(A , m , n );
        for (int index = 0; index < LOOP; index+=gap){
            caffe::Timer _time; 
            _time.Start();
            //mkl_sdnscsr(job, &m, &n, A, &lda, acsr, ja, ia, &info);
            for (int j = 0 ; j < gap; ++j)
                info = Convert(A, m, n, acsr, ja, ia);
            Average_time += _time.MicroSeconds();
            LOG(INFO) << "Iteration " << index << " Convert in " << _time.MicroSeconds()/float(gap) << "us";
        }
        LOG(INFO) << "Average Time for Convert is " << Average_time / LOOP << "us";
        std::ostringstream buffer;
        buffer << "Acsr (" << ia[m] << ") : ";
        const int no_zeros = sizeof(values) / sizeof(values[0]);
        CHECK_EQ( no_zeros, ia[m] );
        for (int i = 0; i < ia[m]; ++i){
            buffer << " " << acsr[i];
            CHECK_EQ( acsr[i] , values[i] );
        }
        LOG(INFO) << buffer.str();  buffer.str("");
        buffer << "rowIndex (" << m << "+1) : ";
        for(int i = 0; i <= m ; ++i){
            CHECK_EQ( rowIndex[i] , ia[i] );
            buffer << " " << ia[i];
        }
        LOG(INFO) << buffer.str();  buffer.str("");
        buffer << "columns (" << ia[m] << ") : ";
        for(int i = 0; i < ia[m] ; ++i){
            CHECK_EQ( columns[i] , ja[i] );
            buffer << " " << ja[i];
        }
        LOG(INFO) << buffer.str();  buffer.str("");
        if(info==0) buffer << "convert is successful ";
        else buffer << "routine is interrupted processing the " << info << "-th row";
        LOG(INFO) << buffer.str();
    }
#else 
    void Sparse_Matrix::TestConvert(){
        LOG(FATAL) << "This func requires MKL; compile with BLAS: mkl.";
    }
#endif

#ifdef USE_MKL
    void Sparse_Matrix::TestMVProduct(const int m,const int n,const double zero_ratio){
        /*
        // m -> rows , n -> cols 
        The mkl_?csrmv routine performs a matrix-vector operation defined as  
          y := alpha*A*x + beta*y  or  y := alpha*A'*x + beta*y,
          where:
            alpha and beta are scalars,
            x and y are vectors,
            A is an m-by-k sparse matrix in the CSR format, A' is the transpose of A.
        */
        CHECK( zero_ratio >= 0 && zero_ratio <= 1 );
        CHECK( m > 0 && n > 0 );
        float *A = new float[m*n];
        const int count_nozeros = GetRandomMatrix(A , m , n , 0 , zero_ratio);
        /*
        int count_nozeros = m*n;
        for (int index = 0; index < m*n; ++index){
            A[index] = (RD()%5000) / 1000.;
            if( RD()%1000 < zero_ratio*1000 ) A[index] = 0;
            //if( A[index] == 0 ) count_nozeros --;
            if( std::abs(A[index])<1e-8 ) count_nozeros --;
        }*/
        LOG(INFO) << "Random Set Matrix A (" << m << " , " << n << ") : " << count_nozeros*1.0/(m*n) << " (force) no-zeros elements" ;
        DisplayMatrix(A, m , n );
        //void mkl_scsrmv(char *transa, int *m, int *k, float *alpha, char *matdescra, float *val, int *indx, int *pntrb, int *pntre, float *x, float *beta, float *y);
        const char transa = 'N'; // If transa = 'N' or 'n', then y := alpha*A*x + beta*y
        // If transa = 'T' or 't' or 'C' or 'c', then y := alpha*A'*x + beta*y,
        // m  INTEGER. Number of rows of the matrix A.
        const int k = n;              //INTEGER. Number of columns of the matrix A.
        const float alpha = 2.0;            // REAL for mkl_scsrmv.
        const char matdescra[6] = {'G','L','N','C','\0','\0'}; //CHARACTER. Array of six elements, specifies properties of the matrix used for operation.
        float *val = new float[count_nozeros]; // Array containing non-zero elements of the matrix A. 
        int *indx = new int[count_nozeros]; //     INTEGER. Array containing the column indices for each non-zero element of the matrix A.Its length is equal to length of the val array.

        int *rowIndex = new int[m+1];
        LOG(INFO) << "Start Convert [MV-Product]";
        caffe::Timer _time; 
        double matrix_time = 0; 
        for (int loop = 0; loop < 10 ; loop++ ){
            _time.Start(); 
            int info = Convert(A, m, k, val, indx, rowIndex);
            matrix_time = _time.MicroSeconds();
            if(info==0) LOG(INFO) << "Convert is successful [MV-Product] , " << matrix_time << " us";
            else LOG(INFO) << "Routine is interrupted processing the " << info << "-th row [MV-Product]";
        }
        int *pntrb = new int[m];
        int *pntre = new int[m];
        memcpy(pntrb, rowIndex, sizeof(int)*m);
        LOG(INFO) << "Prointer B OK ";
        memcpy(pntre, rowIndex+1, sizeof(int)*m);
        LOG(INFO) << "Prointer E OK ";
        for (int i = 0; i < m; i++){
            CHECK_EQ( pntrb[i] , rowIndex[i] );
            CHECK_EQ( pntre[i] , rowIndex[i+1] );
        }
        
        float *x = new float[k]; // Array, DIMENSION at least k if transa = 'N' or 'n' and at least m otherwise. On entry, the array x must contain the vector x. 
        const float beta = 1.7;  // Specifies the scalar beta. 
        float *y = new float[m];
        /* 
        for (int index = 0; index < k; ++index){
            x[index] = (RD()%5000) / 1000.;
        }
        for (int index = 0; index < m; ++index){
            y[index] = (RD()%5000) / 1000.;
        }*/
        const int x_nz = GetRandomMatrix(x , k , 1 , 101);
        const int y_nz = GetRandomMatrix(y , 1 , m , 209);
        LOG(INFO) << "Transa : " << transa << " :: X(" << x_nz << ") Y(" << y_nz << ")";
        LOG(INFO) << "m : " << m << "  , k : " << k;
        LOG(INFO) << "alpha : " << alpha << "  , beta : " << beta;
        LOG(INFO) << "Val : ";
        DisplayMatrix(val, 1, count_nozeros );
        LOG(INFO) << "Index : ";
        DisplayMatrix(indx, 1, count_nozeros );
        LOG(INFO) << "PointerB : ";
        DisplayMatrix(pntrb, 1, m );
        LOG(INFO) << "PointerE : ";
        DisplayMatrix(pntre, 1, m );
        LOG(INFO) << "[MV-Product] Data Prepare Done, Matrix Convert Cost " << matrix_time << " us"; 
        const int LOOP = 30;
        float *temp = new float[m];
        float *XX = new float[k];
        float *YY = new float[m];
        memcpy(XX,x,sizeof(float)*k);
        memcpy(YY,y,sizeof(float)*m);
        vector<double> DenseAVE;
        for (int loop = 0; loop < LOOP; ++loop){
            // Force to calculate Y
            _time.Start();
            for (int j = 0; j < m; ++j){
                temp[j] = beta * y[j];
                for (int i = 0; i < k; ++i){
                    temp[j] += alpha * A[j*k+i] * x[i];
                }
            } 
            //LOG(INFO) << "[Sparse] Loop " << loop << " Done in " << _time.MicroSeconds() << " us";
            double force_time = _time.MicroSeconds();
            _time.Start();
            mkl_scsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
            double sparse_time = _time.MicroSeconds();
            _time.Start();
            caffe_cpu_gemv(CblasNoTrans, m, n, alpha, A, XX, beta, YY);
            double dense_time = _time.MicroSeconds();
            DenseAVE.push_back( dense_time );
            //LOG(INFO) << "[Dense] Loop " << loop << " Done in " << _time.MicroSeconds() << " us";
            LOG(INFO) << "Loop " << loop << " [Force : " << std::setfill(' ') << std::setw(5) << force_time << "]  [Sparse : " << std::setfill(' ') << std::setw(5) << sparse_time << "]  [Dense : " << std::setfill(' ') << std::setw(5) << dense_time << "] us";
            for (int j = 0; j < m; ++j){
                //printf("Temp: %.8f \n, Y : %.8f\n",temp[j],y[j]);
                //CHECK( std::abs(temp[j]-y[j]) <= 1e-5*std::abs(y[j]) ) << j <<   "-th value of [Sparse] Y is wrong. " << temp[j] << "  vs  " << y[j];
                float Sparse_Error = std::max( std::abs(temp[j]-y[j]) / std::abs(y[j]) , std::abs(temp[j]-y[j]) / std::abs(temp[j]) ) ;
                if( Sparse_Error > 3e-5 ) LOG(WARNING) << j << "-th value of [Sparse] Y is wrong. By " << Sparse_Error << " deviation"; 
                //printf("Temp: %.8f \n, Y : %.8f\n",temp[j],y[j]);
                //CHECK( std::abs(temp[j]-YY[j]) <= 1e-5*std::abs(YY[j]) ) << j << "-th value of [Dense] Y is wrong. " << temp[j] << "  vs  " << YY[j];
                Sparse_Error = std::max( std::abs(temp[j]-YY[j]) / std::abs(YY[j]) , std::abs(temp[j]-YY[j]) / std::abs(temp[j]) ) ;
                if( Sparse_Error > 3e-5 ) LOG(WARNING) << j << "-th value of [Dense] Y is wrong. By " << Sparse_Error << " deviation"; 
            }
        }
        sort(DenseAVE.begin(), DenseAVE.end());
        double ave = 0 ;
        int num = 0;
        for (int i = 4; i < LOOP - 4 ; i ++ ){
            ave += DenseAVE[i];
            num ++;
        }
        LOG(INFO) << "[MV] Dense Average Time: " << ave/num << " us";
         
        delete []XX;
        delete []YY;
        delete []A;
        delete []val;
        delete []indx;
        delete []rowIndex;
        delete []pntrb;
        delete []pntre;
        delete []x;
        delete []y;
        delete []temp;
    }
#else
    void Sparse_Matrix::TestMVProduct(const int m, const int n, const double zero_ratio){
        /*
        The since mkl cannot be used, this will only test dense matrix test mv product
          y := alpha*A*x + beta*y  or  y := alpha*A'*x + beta*y,
          where:
            alpha and beta are scalars,
            x and y are vectors,
            A is an m-by-k sparse matrix in the CSR format, A' is the transpose of A.
        */
        CHECK( zero_ratio >= 0 && zero_ratio <= 1 );
        CHECK( m > 0 && n > 0 );
        float *A = new float[m*n];
        const int count_nozeros = GetRandomMatrix(A , m , n , 0 , zero_ratio);
        LOG(INFO) << "Random Set Matrix A (" << m << " , " << n << ") : " << count_nozeros*1.0/(m*n) << " (force) no-zeros elements" ;
        DisplayMatrix(A, m , n );
        const int k = n;
        const float alpha = 3.9;  // Specifies the scalar beta. 
        float *x = new float[k]; // Array, DIMENSION at least k if transa = 'N' or 'n' and at least m otherwise. On entry, the array x must contain the vector x. 
        const float beta = 1.7;  // Specifies the scalar beta. 
        float *y = new float[m];
        GetRandomMatrix(x , k , 1 , 101);
        GetRandomMatrix(y , 1 , m , 209);
        LOG(INFO) << "m : " << m << "  , k : " << k;
        LOG(INFO) << "alpha : " << alpha << "  , beta : " << beta;
        LOG(INFO) << "[MV-Product] Data Prepare Done,.";
        const int LOOP = 30;
        vector<double> DenseAVE;
        caffe::Timer _time; 
        for (int loop = 0; loop < LOOP; ++loop){
            // Force to calculate Y
            _time.Start();
            caffe_cpu_gemv(CblasNoTrans, m, n, alpha, A, x, beta, y);
            double dense_time = _time.MicroSeconds();
            DenseAVE.push_back( dense_time );
            LOG(INFO) << "Loop " << loop << "Dense : " << std::setfill(' ') << std::setw(5) << dense_time << "] us";
        }
        sort(DenseAVE.begin(), DenseAVE.end());
        double ave = 0 ;
        int num = 0;
        for (int i = 4; i < LOOP - 4 ; i ++ ){
            ave += DenseAVE[i];
            num ++;
        }
        LOG(INFO) << "[MV] Dense Average Time: " << ave/num << " us";
         
        delete []A;
        delete []x;
        delete []y;
    }
#endif

#ifdef USE_MKL
    MM_Time Sparse_Matrix::TestMMProduct(const int m, const int k, const int n ,const double zero_ratio){
        /*
        The mkl_?csrmm routine performs a matrix-vector operation defined as  
          C := alpha*A*B + beta*C  or  C := alpha*A'*B + beta*C,
          where:
            alpha and beta are scalars,
            B and C are dense matrices ,
            A is an m-by-k sparse matrix in the CSR format, A' is the transpose of A.
        */
        CHECK( zero_ratio >= 0 && zero_ratio <= 1 );
        //const int m = k + 10;
        CHECK( m > 0 && k > 0 );
        float *A = new float[m*k];
        const int count_nozeros = GetRandomMatrix(A , m , k , 0 , zero_ratio);
        /*
        int count_nozeros = m*n;
        for (int index = 0; index < m*n; ++index){
            A[index] = (RD()%5000) / 1000.;
            if( RD()%1000 < zero_ratio*1000 ) A[index] = 0;
            //if( A[index] == 0 ) count_nozeros --;
            if( std::abs(A[index])<1e-8 ) count_nozeros --;
        }*/
        LOG(INFO) << "Random Set Matrix A (" << m << " , " << k << ") : " << count_nozeros*1.0/(m*k) << " (force) no-zeros elements" ;
        DisplayMatrix(A, m , k );
        //void mkl_scsrmv(char *transa, int *m, int *k, float *alpha, char *matdescra, float *val, int *indx, int *pntrb, int *pntre, float *x, float *beta, float *y);
        const char transa = 'N'; // If transa = 'N' or 'n', then y := alpha*A*x + beta*y
        // If transa = 'T' or 't' or 'C' or 'c', then y := alpha*A'*x + beta*y,
        // m  INTEGER. Number of rows of the matrix A.
        //const int k = n;              //INTEGER. Number of columns of the matrix A.
        const float alpha = 2.0;            // REAL for mkl_scsrmv.
        const char matdescra[6] = {'G','L','N','C','\0','\0'}; //CHARACTER. Array of six elements, specifies properties of the matrix used for operation.
        float *val = new float[count_nozeros]; // Array containing non-zero elements of the matrix A. 
        int *indx = new int[count_nozeros]; //     INTEGER. Array containing the column indices for each non-zero element of the matrix A.Its length is equal to length of the val array.

        int *rowIndex = new int[m+1];
        LOG(INFO) << "Start Convert [MV-Product]";
        caffe::Timer _time; 
        double matrix_time = 0; 
        for (int loop = 0; loop < 10 ; loop++ ){
            _time.Start(); 
            int info = Convert(A, m, k, val, indx, rowIndex);
            matrix_time = _time.MicroSeconds();
            if(info==0) LOG(INFO) << "Convert is successful [MV-Product] , " << matrix_time << " us";
            else LOG(INFO) << "Routine is interrupted processing the " << info << "-th row [MV-Product]";
        }
        int *pntrb = new int[m];
        int *pntre = new int[m];
        memcpy(pntrb, rowIndex, sizeof(int)*m);
        LOG(INFO) << "Prointer B OK ";
        memcpy(pntre, rowIndex+1, sizeof(int)*m);
        LOG(INFO) << "Prointer E OK ";
        for (int i = 0; i < m; i++){
            CHECK_EQ( pntrb[i] , rowIndex[i] );
            CHECK_EQ( pntre[i] , rowIndex[i+1] );
        }
        
        //const int n = k + 1;
        float *B = new float[k*n]; // Array, DIMENSION at least k if transa = 'N' or 'n' and at least m otherwise. On entry, the array x must contain the vector x. 
        const float beta = 1.7;  // Specifies the scalar beta. 
        float *C = new float[m*n];
        const int x_nz = GetRandomMatrix(B , k , n , 121);
        const int y_nz = GetRandomMatrix(C , m , n , 207);
        LOG(INFO) << "Transa : " << transa << " :: X(" << x_nz << ") Y(" << y_nz << ")";
        LOG(INFO) << "m : " << m << "  , k : " << k;
        LOG(INFO) << "alpha : " << alpha << "  , beta : " << beta;
        LOG(INFO) << "Val : ";
        DisplayMatrix(val, 1, count_nozeros );
        LOG(INFO) << "Index : ";
        DisplayMatrix(indx, 1, count_nozeros );
        LOG(INFO) << "PointerB : ";
        DisplayMatrix(pntrb, 1, m );
        LOG(INFO) << "PointerE : ";
        DisplayMatrix(pntre, 1, m );
        LOG(INFO) << "[MV-Product] Data Prepare Done, Matrix Convert Cost " << matrix_time << " us"; 
        const int LOOP = 30;
        const int ldb = n;
        const int ldc = n;
        float *XX = new float[k*n];
        float *YY = new float[m*n];
        memcpy(XX,B,sizeof(float)*k*n);
        memcpy(YY,C,sizeof(float)*m*n);
        vector<double> DenseAVE;
        for (int loop = 0; loop < LOOP; ++loop){
            // Force to calculate Y
            _time.Start();
            mkl_scsrmm(&transa, &m, &n, &k, &alpha, matdescra, val, indx, pntrb, pntre, B, &ldb, &beta, C, &ldc);
            //double sparse_time = _time.MicroSeconds();
            _time.Start();
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, XX, beta, YY);
            double dense_time = _time.MicroSeconds();
            //LOG(INFO) << "Loop " << loop << " [Sparse : " << std::setfill(' ') << std::setw(5) << sparse_time << "]  [Dense : " << std::setfill(' ') << std::setw(5) << dense_time << "] us";
            DenseAVE.push_back(dense_time);
            for (int j = 0; j < m*n; ++j){
                double Sparse_Error = std::max( std::abs(C[j]-YY[j]) / std::abs(YY[j]) , std::abs(C[j]-YY[j]) / std::abs(C[j]) ) ;
                if( Sparse_Error > 3e-5 ) LOG(WARNING) << j << "-th value of [Dense] vs [Sparse] is wrong. By " << Sparse_Error << " deviation"; 
            }
        }
        sort(DenseAVE.begin(), DenseAVE.end());
        double ave = 0 ;
        int num = 0;
        for (int i = 4; i < LOOP - 4 ; i ++ ){
            ave += DenseAVE[i];
            num ++;
        }
        LOG(INFO) << "Dense Average Time: " << ave/num << " us";
        delete []XX;
        delete []YY;
        delete []A;
        delete []val;
        delete []indx;
        delete []rowIndex;
        delete []pntrb;
        delete []pntre;
        delete []B;
        delete []C;
        return MM_Time(m, n, k, ave/num);
    }
#else
    MM_Time Sparse_Matrix::TestMMProduct(const int m, const int k, const int n ,const double zero_ratio){
        /*
        The since mkl cannot be used, this will only test dense matrix test mm product
          C := alpha*A*B + beta*C  or  C := alpha*A'*B + beta*C,
          where:
            alpha and beta are scalars,
            B and C are dense matrices ,
            A is an m-by-k sparse matrix in the CSR format, A' is the transpose of A.
        */
        CHECK( zero_ratio >= 0 && zero_ratio <= 1 ) << " not " << zero_ratio;
        //const int m = k + 10;
        CHECK( m > 0 && k > 0 );
        float *A = new float[m*k];
        const int count_nozeros = GetRandomMatrix(A , m , k , 0 , zero_ratio);
        LOG(INFO) << "Random Set Matrix A (" << m << " , " << k << ") : " << count_nozeros*1.0/(m*k) << " (force) no-zeros elements" ;
        DisplayMatrix(A, m , k );
        //void mkl_scsrmv(char *transa, int *m, int *k, float *alpha, char *matdescra, float *val, int *indx, int *pntrb, int *pntre, float *x, float *beta, float *y);
        //const int k = n;              //INTEGER. Number of columns of the matrix A.
        const float alpha = 2.0;            // REAL for mkl_scsrmv.
        //const int n = k + 1;
        float *B = new float[k*n]; // Array, DIMENSION at least k if transa = 'N' or 'n' and at least m otherwise. On entry, the array x must contain the vector x. 
        const float beta = 1.7;  // Specifies the scalar beta. 
        float *C = new float[m*n];
        GetRandomMatrix(B , k , n , 121);
        GetRandomMatrix(C , m , n , 207);
        LOG(INFO) << "m : " << m << "  , k : " << k << " , n : " << n;
        LOG(INFO) << "alpha : " << alpha << "  , beta : " << beta;
        //LOG(INFO) << "[MV-Product] Data Prepare Done";
        const int LOOP = 30;
        vector<double> DenseAVE;
        caffe::Timer _time; 
        for (int loop = 0; loop < LOOP; ++loop){
            // Force to calculate Y
            _time.Start();
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, B, beta, C);
            double dense_time = _time.MicroSeconds();
            //LOG(INFO) << "Loop " << loop << "[Dense : " << std::setfill(' ') << std::setw(5) << dense_time << "] us";
            DenseAVE.push_back(dense_time);
        }
        sort(DenseAVE.begin(), DenseAVE.end());
        LOG(INFO) << "FOR CHECK >>>>  m : " << m << " n : " << n << " k : " << k;
        double ave = 0 ;
        int num = 0;
        for (int i = 4; i < LOOP - 4 ; i ++ ){
            ave += DenseAVE[i];
            num ++;
        }
        LOG(INFO) << "Dense Average Time: " << ave/num << " us";
        delete []A;
        delete []B;
        delete []C;
        return MM_Time(m, n, k, ave/num);
    }
#endif

}   // namespace caffe
