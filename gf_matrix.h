#include "gf_int.h"

#include <cstdio>

constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

template <int M, int N, int BITS>
class gf_matrix
{
public:
    using data_t = gf_int<BITS>;

    data_t data[M][N];

    gf_matrix() = default;
};

template <int M, int N, int K, int BITS>
__global__ void gf_mul_matrix(
    const gf_matrix<M, N, BITS> *_A,
    const gf_matrix<N, K, BITS> *_B,
    gf_matrix<M, K, BITS> *_C)
{
    const auto &A = *_A;
	const auto &B = *_B;
	auto &C = *_C;

	int begin_x = blockIdx.x * blockDim.x + threadIdx.x;
	int begin_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_x = gridDim.x * blockDim.x;
	int stride_y = gridDim.y * blockDim.y;

    for (int i = begin_x; i < M; i += stride_x) {
		for (int k = begin_y; k < K; k += stride_y) {
			gf_int<BITS> sum(0);
			for (int j = 0; j < N; ++j) {
				sum += A[i][j] * B[j][k];
			}
			C[i][k] = sum;
		}
	}
}



template <int M, int BITS>
using gf_square = gf_matrix<M, M, BITS>;



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



template <typename T, int M, int N, int Mp, int Np>
void shared_load(T dest[M][N], T src[Mp][Np], int begin_x, int begin_y)
{
    dest[threadIdx.x][threadIdx.y] = src[begin_x + threadIdx.x][begin_y + threadIdx.y];
    __syncthreads();
}

template <typename T, int M, int N, int Mp, int Np>
void shared_store(T src[M][N], T dest[Mp][Np], int begin_x, int begin_y)
{
    dest[begin_x + threadIdx.x][begin_y + threadIdx.y] = src[threadIdx.x][threadIdx.y];
    __syncthreads();
}

template <typename T, int M, int N, int K>
void shared_mul(T A[M][N], T B[N][K], T C[M][K])
{
    T sum(0);
    for (int j = 0; j < N; ++j) {
        sum += A[threadIdx.x][j] * B[j][threadIdx.y];
    }
    C[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();
}

template <typename T, int N>
void shared_inverse(T A[N][N], T left[N][N])
{
    left[threadIdx.x][threadIdx.y] = (threadIdx.x == threadIdx.y) ? T(1) : T(0);
    __syncthreads();

    for (int pivot_idx = 0; pivot_idx < N; ++pivot_idx) {
        T pivot_inv = A[pivot_idx][pivot_idx].inverse();

        T coeff = (threadIdx.x == pivot_idx) ? T(0) : (pivot_inv * A[threadIdx.x][pivot_idx]);
        __syncthreads();

        A[threadIdx.x][threadIdx.y] += coeff * A[pivot_idx][threadIdx.y];
        left[threadIdx.x][threadIdx.y] += coeff * left[pivot_idx][threadIdx.y];

        __syncthreads();
    }
}



template <int M, int BITS>
__global__ void block_elim_round(
    gf_square<M, BITS> *_squareA,
    gf_square<M, BITS> *_squareB,
    const int idx_pivot_block)
{
    auto &dataA = (*_squareA).data;
    auto &dataB = (*_squareB).data;

    int begin_x = (blockIdx.x + idx_pivot_block) * blockDim.x;
    int stride_x = gridDim.x * blockDim.x;
    int begin_y = idx_pivot_block * blockDim.y;

    for (int i = begin_x; i < M; i += stride_x) {
        __shared__ pivot_block[BLOCK_DIM_X][BLOCK_DIM_Y];
        __shared__ coeff_block[BLOCK_DIM_X][BLOCK_DIM_Y];

        shared_load(pivot_block, dataA, i, begin_y);
        shared_inverse(pivot_block, coeff_block);

        for (int j = begin_y + blockDim.y; j < M; j += blockDim.y) {
            __shared__ triv_block[BLOCK_DIM_X][BLOCK_DIM_Y];
            shared_load(triv_block, dataA, i, j);

            __shared__ result_block[BLOCK_DIM_X][BLOCK_DIM_Y];
            shared_mul(triv_block, coeff_block, result_block);
            
            shared_store(result_block, dataA, i, j);
        }
    }
}