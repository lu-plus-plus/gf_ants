
#include "gf_matrix.h"
#include "cuder.h"

constexpr int M = 20;
constexpr int N = M*2;

using gf_int_t = gf_int<CURRENT_BITS>;
using matrix_t = gf_matrix<CURRENT_BITS, M, N>;



matrix_t h_mat;

int main(void)
{
	cuder<matrix_t> d_mat_ptr(make_cuder<matrix_t>());
	cuder<gf_int_t> d_coeff_ptr(make_cuder<gf_int_t>(M));

	for (uint32_t i = 0; i < M; ++i) {
		for (uint32_t j = i; j < M; ++j) {
			h_mat.data[i][j] = matrix_t::data_t(i + j + 1);
		}

		h_mat.data[i][i+M] = matrix_t::data_t(1);
	}
	/*for (uint32_t i = 0; i < M; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			std::cout << std::hex << h_mat.data[i][j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/
	
	cudaMemcpy(d_mat_ptr.toKernel(), &h_mat, sizeof(h_mat), cudaMemcpyHostToDevice);

	check_inverse<<<dim3(16, 16), dim3(BLOCK_DIM_X, BLOCK_DIM_Y, CURRENT_BITS)>>>
		(d_mat_ptr.toKernel());

	for (int num_pivot = 0; num_pivot < M; ++num_pivot) {
		calcu_row_coeffs<<<dim3(16, 16), dim3(BLOCK_DIM_X, BLOCK_DIM_Y, CURRENT_BITS)>>>
			(d_mat_ptr.toKernel(), d_coeff_ptr.toKernel(), num_pivot);
		
		eliminate_rows<<<dim3(16, 16), dim3(BLOCK_DIM_X, BLOCK_DIM_Y, CURRENT_BITS)>>>
			(d_mat_ptr.toKernel(), d_coeff_ptr.toKernel(), num_pivot);
	}

	normalize_by_pivots<<<dim3(16, 16), dim3(BLOCK_DIM_X, BLOCK_DIM_Y, CURRENT_BITS)>>>
		(d_mat_ptr.toKernel());
	
	cudaMemcpy(&h_mat, d_mat_ptr.toKernel(), sizeof(h_mat), cudaMemcpyDeviceToHost);
	for (uint32_t i = 0; i < M; ++i) {
		for (uint32_t j = M; j < N; ++j) {
			std::cout << std::hex << h_mat.data[i][j] << ' ';
		}
		std::cout << std::endl;
	}

	return 0;
}