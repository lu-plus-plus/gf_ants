
#include "gf_matrix.h"
#include "cuder.h"

constexpr bool PRINT_VERBOSE = 0;
constexpr bool PRINT_INITIAL_VALUE = 0;
constexpr bool PRINT_RESULT = 0;



constexpr int M = 2048;

constexpr int BITS = 8;

using gf_int_t = gf_int<BITS>;
using square_t = gf_square<M, BITS>;



square_t h_A;
square_t h_B;

int main(void)
{
	cuder<square_t> d_A_ptr(make_cuder<square_t>());
	cuder<square_t> d_B_ptr(make_cuder<square_t>());
	cuder<gf_int_t> d_coeff_ptr(make_cuder<gf_int_t>(M));

	for (uint32_t i = 0; i < M; ++i) {
		for (uint32_t j = i; j < M; ++j) {
			h_A.data[i][j] = gf_int_t(i + j + 1);
		}

		h_B.data[i][i] = gf_int_t(1);
	}

	if (PRINT_INITIAL_VALUE) {
		for (uint32_t i = 0; i < M; ++i) {
			for (uint32_t j = 0; j < M; ++j) {
				std::cout << std::hex << h_A.data[i][j] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	
	cudaMemcpy(d_A_ptr.toKernel(), &h_A, sizeof(h_A), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_ptr.toKernel(), &h_B, sizeof(h_B), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	if (PRINT_VERBOSE) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}

	for (int num_pivot = 0; num_pivot < M; ++num_pivot) {
		Calcu_Row_Coeffs<<<dim3(16, 16), dim3(16, 16)>>>
			(d_A_ptr.toKernel(), d_coeff_ptr.toKernel(), num_pivot);
		cudaDeviceSynchronize();

		Eliminate_Rows<<<dim3(16, 16), dim3(16, 16)>>>
			(d_A_ptr.toKernel(), d_B_ptr.toKernel(), d_coeff_ptr.toKernel(), num_pivot);
		cudaDeviceSynchronize();

		if (PRINT_VERBOSE) {
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime, start, stop);
			std::cout << "Round " << num_pivot << ": " << (elapsedTime/1000) << " s" << std::endl;
		}	
	}

	Normalize_By_Pivots<<<dim3(16, 16), dim3(16, 16)>>>
		(d_A_ptr.toKernel(), d_B_ptr.toKernel());
	cudaDeviceSynchronize();
	
	cudaMemcpy(&h_A, d_A_ptr.toKernel(), sizeof(h_A), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_B, d_B_ptr.toKernel(), sizeof(h_B), cudaMemcpyDeviceToHost);

	if (PRINT_RESULT) {
		for (uint32_t i = 0; i < M; ++i) {
			for (uint32_t j = 0; j < M; ++j) {
				std::cout << std::hex << h_B.data[i][j] << ' ';
			}
			std::cout << std::endl;
		}
	}

	return 0;
}