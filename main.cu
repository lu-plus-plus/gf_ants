
#include "gf_matrix.h"
#include "cuder.h"

// constexpr bool PRINT_VERBOSE = 0;
constexpr bool PRINT_INITIAL_VALUE = 0;
constexpr bool PRINT_RESULT = 0;



constexpr int APP_DIM = 1024;

constexpr int APP_BITS = 8;

using gf_int_t = gf_int<APP_BITS>;
using square_t = gf_square<APP_DIM, APP_BITS>;



int main(void)
{
	auto h_A_ptr = new square_t();
	auto h_B_ptr = new square_t();
	square_t &h_A = *h_A_ptr;
	square_t &h_B = *h_B_ptr;

	cuder<square_t> d_A_ptr(make_cuder<square_t>());
	cuder<square_t> d_B_ptr(make_cuder<square_t>());
	cuder<gf_int_t> d_coeff_ptr(make_cuder<gf_int_t>(APP_DIM));

	for (uint32_t i = 0; i < APP_DIM; ++i) {
		for (uint32_t j = i; j < APP_DIM; ++j) {
			h_A.data[i][j] = gf_int_t(i + j + 1);
		}

		h_B.data[i][i] = gf_int_t(1);
	}

	if (PRINT_INITIAL_VALUE) {
		for (uint32_t i = 0; i < APP_DIM; ++i) {
			for (uint32_t j = 0; j < APP_DIM; ++j) {
				std::cout << std::hex << h_A.data[i][j] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	
	cudaMemcpy(d_A_ptr.toKernel(), &h_A, sizeof(h_A), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_ptr.toKernel(), &h_B, sizeof(h_B), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int num_pivot = 0; num_pivot < APP_DIM; ++num_pivot) {
		Calcu_Row_Coeffs<<<dim3(16, 16), dim3(16, 16)>>>
			(d_A_ptr.toKernel(), d_coeff_ptr.toKernel(), num_pivot);
		cudaDeviceSynchronize();

		Eliminate_Rows<<<dim3(16, 16), dim3(16, 16)>>>
			(d_A_ptr.toKernel(), d_B_ptr.toKernel(), d_coeff_ptr.toKernel(), num_pivot);
		cudaDeviceSynchronize();
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "Total time: " << (elapsedTime/1000) << " s" << std::endl;

	Normalize_By_Pivots<<<dim3(16, 16), dim3(16, 16)>>>
		(d_A_ptr.toKernel(), d_B_ptr.toKernel());
	cudaDeviceSynchronize();
	
	cudaMemcpy(&h_A, d_A_ptr.toKernel(), sizeof(h_A), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_B, d_B_ptr.toKernel(), sizeof(h_B), cudaMemcpyDeviceToHost);

	if (PRINT_RESULT) {
		for (uint32_t i = 0; i < APP_DIM; ++i) {
			for (uint32_t j = 0; j < APP_DIM; ++j) {
				std::cout << std::hex << h_B.data[i][j] << ' ';
			}
			std::cout << std::endl;
		}
	}

	return 0;
}