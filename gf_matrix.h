#include "gf_int.h"



/* ************************************************** */
/* General Matrix Type */

template <typename T, int M, int N>
class gf_matrix
{
public:
    T data[M][N];
};

template <typename T, int M>
using gf_square = gf_matrix<T, M, M>;



/*
template <int M, int BITS>
__global__ void Calcu_Row_Coeffs(
    const gf_square<M, BITS> *_squareA,
    gf_int<BITS> coeff[],
    const int num_pivot)
{
    const auto &data = (*_squareA).data;

    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadsPerGrid = threadsPerBlock * gridDim.x * gridDim.y;
    int begin = (blockIdx.x * gridDim.y + blockIdx.y) * threadsPerBlock
        + threadIdx.x * blockDim.y + threadIdx.y;
    int stride = threadsPerGrid;

    const auto pivot_inv = data[num_pivot][num_pivot].inverse();
    for (int i = begin; i < M; i += stride) {
        coeff[i] = data[i][num_pivot] * pivot_inv;
    }
    coeff[num_pivot] = 0;
}

template <int M, int BITS>
__global__ void Eliminate_Rows(
    gf_square<M, BITS> *_squareA,
    gf_square<M, BITS> *_squareB,
    const gf_int<BITS> coeff[],
    const int num_pivot)
{
    auto &dataA = (*_squareA).data;
    auto &dataB = (*_squareB).data;

    int begin_x = blockIdx.x * blockDim.x + threadIdx.x;
    int begin_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = begin_x; i < M; i += stride_x) {
        for (int j = begin_y + num_pivot; j < M; j += stride_y) {
            dataA[i][j] += coeff[i] * dataA[num_pivot][j];
        }
    }

    for (int i = begin_x; i < M; i += stride_x) {
        for (int j = begin_y; j <= num_pivot; j += stride_y) {
            dataB[i][j] += coeff[i] * dataB[num_pivot][j];
        }
    }
}

template <int M, int BITS>
__global__ void Normalize_By_Pivots(
    const gf_square<M, BITS> *_squareA,
    gf_square<M, BITS> *_squareB)
{
    const auto &dataA = (*_squareA).data;
    auto &dataB = (*_squareB).data;

    int begin_x = blockIdx.x * blockDim.x + threadIdx.x;
    int begin_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = begin_x; i < M; i += stride_x) {
        auto pivot_inv = dataA[i][i].inverse();
        
        for (int j = begin_y; j < M; j += stride_y) {
            dataB[i][j] *= pivot_inv;
        }
    }
}
*/



constexpr int BLOCK_DIM = 32;

constexpr int GRID_DIM_X = 128;



template <typename T, int N, int Np>
__device__ inline
void shared_load(T dest[N][N], const T src[][Np], const int begin_y, const int begin_x)
{
    dest[threadIdx.y][threadIdx.x] = src[begin_y + threadIdx.y][begin_x + threadIdx.x];
    __syncthreads();
}

template <typename T, int N, int Np>
__device__ inline
void shared_store(T dest[][Np], const int begin_y, const int begin_x, const T src[N][N])
{
    dest[begin_y + threadIdx.y][begin_x + threadIdx.x] = src[threadIdx.y][threadIdx.x];
    __syncthreads();
}

template <typename T, int N>
__device__ inline
void shared_copy(T dest[N][N], const T src[N][N])
{
    dest[threadIdx.y][threadIdx.x] = src[threadIdx.y][threadIdx.x];
    __syncthreads();
}

template <typename T, int N>
__device__ inline
void shared_identify(T data[N][N])
{
    data[threadIdx.y][threadIdx.x] = (threadIdx.y == threadIdx.x) ? T(1) : T(0);
    __syncthreads();
}

template <typename T, int N>
__device__ inline
void shared_mul(const T A[N][N], const T B[N][N], T C[N][N])
{
    T sum(0);
    for (int j = 0; j < N; ++j) {
        sum += A[threadIdx.y][j] * B[j][threadIdx.x];
    }
    C[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
}

template <typename T, int N>
__device__ inline
void shared_inverse(const T const_A[N][N], T B[N][N])
{
    __shared__ T A[BLOCK_DIM][BLOCK_DIM];
    shared_copy(A, const_A);
    shared_identify(B);

    for (int pivot_idx = 0; pivot_idx < N; ++pivot_idx) {
        if (threadIdx.y != pivot_idx) {
            T coeff = A[pivot_idx][pivot_idx].inverse() * A[threadIdx.y][pivot_idx];

            A[threadIdx.y][threadIdx.x] += coeff * A[pivot_idx][threadIdx.x];
            B[threadIdx.y][threadIdx.x] += coeff * B[pivot_idx][threadIdx.x];
        }
        __syncthreads();
    }

    B[threadIdx.y][threadIdx.x] *= A[threadIdx.y][threadIdx.y].inverse();
    __syncthreads();
}

template <typename T, int N>
__device__ inline
void shared_add(T A[N][N], const T B[N][N])
{
    A[threadIdx.y][threadIdx.x] += B[threadIdx.y][threadIdx.x];
    __syncthreads();
}



template <typename T, int M>
__global__ void shared_op_test(
    gf_square<T, M> *_squareA,
    gf_square<T, M> *_squareB)
{
    auto &dataA = (*_squareA).data;
    auto &dataB = (*_squareB).data;

    __shared__ T A[BLOCK_DIM][BLOCK_DIM];
    __shared__ T B[BLOCK_DIM][BLOCK_DIM];

    shared_load(A, dataB, 0, 0);
    shared_copy(B, A);
    shared_load(A, dataA, 0, 0);

    B[threadIdx.y][threadIdx.x] = T(0x7);
    __syncthreads();
    shared_inverse(A, B);

    __shared__ T C[BLOCK_DIM][BLOCK_DIM];
    shared_mul(A, B, C);

    shared_store(dataB, 0, 0, C);
}



template <typename T, int M>
__global__ void elimination_round(
    gf_square<T, M> *_squareA,
    gf_square<T, M> *_squareB,
    const int pivot_block_idx)
{
    auto &dataA = (*_squareA).data;
    auto &dataB = (*_squareB).data;

    const int block_begin = blockIdx.x;
    const int block_end = M / BLOCK_DIM;
    const int block_stride = gridDim.x;

    __shared__ T pivot_block[BLOCK_DIM][BLOCK_DIM];
    shared_load(pivot_block, dataA, pivot_block_idx * blockDim.y, pivot_block_idx * blockDim.x);
    __shared__ T pivot_inverse[BLOCK_DIM][BLOCK_DIM];
    shared_inverse(pivot_block, pivot_inverse);

    // All the block rows
    for (int block_row_idx = block_begin; block_row_idx < block_end; block_row_idx += block_stride) {
        
        // The pivot-block row is constant in each round
        if (block_row_idx == pivot_block_idx)
            continue;
        
        // Name the pivot block as P, the counterpart in this row as C.
        // C + coeff * P = 0 should be satisfied.
        // 
        // coeff = inv(P) * C
        __shared__ T counterpart[BLOCK_DIM][BLOCK_DIM];
        shared_load(counterpart, dataA, block_row_idx * blockDim.y, pivot_block_idx * blockDim.x);
        __shared__ T coeff[BLOCK_DIM][BLOCK_DIM];
        shared_mul(counterpart, pivot_inverse, coeff);

        __shared__ T base_along_pivot[BLOCK_DIM][BLOCK_DIM];
        __shared__ T addition[BLOCK_DIM][BLOCK_DIM];
        for (int block_col_idx = pivot_block_idx; block_col_idx < block_end; ++block_col_idx) {
            shared_load(base_along_pivot, dataA, pivot_block_idx * blockDim.y, block_col_idx * blockDim.x);
            shared_mul(coeff, base_along_pivot, addition);
            
            shared_load(counterpart, dataA, block_row_idx * blockDim.y, block_col_idx * blockDim.x);
            shared_add(counterpart, addition);
            shared_store(dataA, block_row_idx * blockDim.y, block_col_idx * blockDim.x, counterpart);
        }
        for (int block_col_idx = 0; block_col_idx <= pivot_block_idx; ++block_col_idx) {
            shared_load(base_along_pivot, dataB, pivot_block_idx * blockDim.y, block_col_idx * blockDim.x);
            shared_mul(coeff, base_along_pivot, addition);
            
            shared_load(counterpart, dataB, block_row_idx * blockDim.y, block_col_idx * blockDim.x);
            shared_add(counterpart, addition);
            shared_store(dataB, block_row_idx * blockDim.y, block_col_idx * blockDim.x, counterpart);
        }

    }
}



template <typename T, int M>
__global__ void normalize_by_pivots(
    const gf_square<T, M> *_squareA,
    gf_square<T, M> *_squareB)
{
    const auto &dataA = (*_squareA).data;
    auto &dataB = (*_squareB).data;

    const int block_begin = blockIdx.x;
    const int block_end = M / BLOCK_DIM;
    const int block_stride = gridDim.x;

    for (int pivot_block_idx = block_begin; pivot_block_idx < block_end; pivot_block_idx += block_stride) {
        __shared__ T pivot_block[BLOCK_DIM][BLOCK_DIM];
        shared_load(pivot_block, dataA, pivot_block_idx * blockDim.y, pivot_block_idx * blockDim.x);
        __shared__ T pivot_inverse[BLOCK_DIM][BLOCK_DIM];
        shared_inverse(pivot_block, pivot_inverse);

        __shared__ T base[BLOCK_DIM][BLOCK_DIM];
        __shared__ T result[BLOCK_DIM][BLOCK_DIM];
        for (int block_col_idx = 0; block_col_idx < block_end; ++block_col_idx) {
            shared_load(base, dataB, pivot_block_idx * blockDim.y, block_col_idx * blockDim.x);
            shared_mul(pivot_inverse, base, result);
            shared_store(dataB, pivot_block_idx * blockDim.y, block_col_idx * blockDim.x, result);
        }
    }
}