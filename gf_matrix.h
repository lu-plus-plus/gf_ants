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

template <typename T, int N, int Np>
__device__ inline
void global_add(T dest[][Np], const int begin_y, const int begin_x, const T src[N][N])
{
    dest[begin_y + threadIdx.y][begin_x + threadIdx.x] += src[threadIdx.y][threadIdx.x];
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
            
            global_add(dataA, block_row_idx * blockDim.y, block_col_idx * blockDim.x, addition);
        }
        for (int block_col_idx = 0; block_col_idx <= pivot_block_idx; ++block_col_idx) {
            shared_load(base_along_pivot, dataB, pivot_block_idx * blockDim.y, block_col_idx * blockDim.x);
            shared_mul(coeff, base_along_pivot, addition);
            
            global_add(dataB, block_row_idx * blockDim.y, block_col_idx * blockDim.x, addition);
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