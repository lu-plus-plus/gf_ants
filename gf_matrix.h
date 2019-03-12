#include "gf_int.h"

#include <cstdio>

template <int BITS, int M, int N>
class gf_matrix;

template <int BITS, int M, int N>
__global__ void Inverse_Precheck(gf_matrix<BITS,M,N> *_mat) {
    const auto &data = (*_mat).data;

    int threadsPerBlock = (blockDim.x * blockDim.y);
    int absBlockIdx = blockIdx.x * gridDim.y + blockIdx.y;
    int absThreadIdx = absBlockIdx * threadsPerBlock
        + threadIdx.x * blockDim.y + threadIdx.y;
    int absThreadsNumber = (gridDim.x * gridDim.y) * threadsPerBlock;

    if (M*2 != N) {
        if (absThreadIdx == 0 && threadIdx.z == 0)
            printf("Error: The matrix must be of form like [Square, I]\n");
        return;
    }

    for (int i = absThreadIdx; i < M; i += absThreadsNumber) {
        if (data[i][i].value() == 0 && threadIdx.z == 0) {
            printf("Error: Pivot[%d][%d] is 0.\n", i, i);
        }
    }
}

template <int BITS, int M, int N>
__global__ void Calcu_Row_Coeffs(gf_matrix<BITS,M,N> *_mat,
    gf_int<BITS> coeff[], const int num_pivot) {
    auto &data = (*_mat).data;

    int begin_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;

    auto pivot_inv = data[num_pivot][num_pivot].inverse();
    for (int i = begin_x; i < M; i += stride_x) {
        coeff[i] = data[i][num_pivot] * pivot_inv;
    }

    coeff[num_pivot] = 0;
}

template <int BITS, int M, int N>
__global__ void Eliminate_Rows(gf_matrix<BITS,M,N> *_mat,
    gf_int<BITS> coeff[], const int num_pivot) {
    auto &data = (*_mat).data;

    int begin_x = blockIdx.x * blockDim.x + threadIdx.x;
    int begin_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = begin_x; i < M; i += stride_x) {
        for (int j = begin_y; j < N; j += stride_y) {
            data[i][j] += coeff[i] * data[num_pivot][j];
            __syncthreads();
        }
        __syncthreads();
    }
}

template <int BITS, int M, int N>
__global__ void Normalize_By_Pivots(gf_matrix<BITS,M,N> *_mat) {
    auto &data = (*_mat).data;

    int begin_x = blockIdx.x * blockDim.x + threadIdx.x;
    int begin_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = begin_x; i < M; i += stride_x) {
        auto pivot_inv = data[i][i].inverse();
        for (int j = M + begin_y; j < N; j += stride_y) {
            data[i][j] *= pivot_inv;
            __syncthreads();
        }
        __syncthreads();
    }
}

template <int BITS, int M, int N>
class gf_matrix
{
public:
    using data_t = gf_int<BITS>;

    data_t data[M][N];

    gf_matrix() = default;
};
