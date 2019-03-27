
#include "gf_matrix.h"
#include "cuder.h"

// constexpr bool PRINT_VERBOSE = 0;
// constexpr bool PRINT_INITIAL_VALUE = 0;
// constexpr bool PRINT_RESULT = 0;



constexpr int M = 1024;

constexpr int BITS = 8;

using gf_int_t = gf_int<BITS>;
using square_t = gf_square<gf_int_t, M>;



// Allocate static memory on host
static square_t h_A;
static square_t h_B;
static square_t h_C;



void gaussian_elimination(square_t *d_A_ptr, square_t *d_B_ptr)
{
	dim3 grid(GRID_DIM_X);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	
	for (int pivot_block_idx = 0; pivot_block_idx < (M / BLOCK_DIM); ++pivot_block_idx) {
		elimination_round<<<grid, block>>>(d_A_ptr, d_B_ptr, pivot_block_idx);
		cudaDeviceSynchronize();
	}

	normalize_by_pivots<<<grid, block>>>(d_A_ptr, d_B_ptr);
	cudaDeviceSynchronize();
}



void init_A(square_t *obj) {
	for (uint32_t i = 0; i < M; ++i) {
		for (uint32_t j = i; j < M; ++j) {
			obj->data[i][j] = gf_int_t(i + j + 1);
		}
	}
}

void init_B(square_t *obj) {
	for (uint32_t i = 0; i < M; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			obj->data[i][j] = gf_int_t(i + (M + j) + 1);
		}
	}
}

void init_D(square_t *obj) {
	for (uint32_t i = 0; i < M; ++i) {
		for (uint32_t j = i; j < M; ++j) {
			obj->data[i][j] = gf_int_t(2*M + i + j + 1);
		}
	}
}

void host_identify(square_t *obj) {
	for (uint32_t i = 0; i < M; ++i) {
		obj->data[i][i] = gf_int_t(1);
	}
}



int main(void)
{
	if (M % BLOCK_DIM != 0) {
		std::cout << "This square matrix isn't block-size-aligned." << std::endl;
		throw std::exception();
	}

	try {
		cuder<square_t> buf_1(&h_A, make_remote<square_t>());
		cuder<square_t> buf_2(&h_B, make_remote<square_t>());
		cuder<square_t> buf_3(&h_C, make_remote<square_t>());

		buf_1.init(init_A);
		buf_2.init(host_identify);
		gaussian_elimination(buf_1.toKernel(), buf_2.toKernel());
		buf_2.write_disk("A_inv.bin");
		// buf_1 = undefined
		// buf_2 = A_inv
		// buf_3 = undefined

		buf_1.init(init_D);
		buf_3.init(host_identify);
		gaussian_elimination(buf_1.toKernel(), buf_3.toKernel());
		buf_3.write_disk("D_inv.bin");
		// buf_1 = undefined
		// buf_2 = A_inv
		// buf_3 = D_inv

		buf_1.init(init_B);
		gf_matrix_mul<<<dim3(16,16), dim3(BLOCK_DIM, BLOCK_DIM)>>>
			(buf_2.toKernel(), buf_1.toKernel(), buf_3.toKernel());
		cudaDeviceSynchronize();
		// buf_1 = B
		// buf_2 = A_inv
		// buf_3 = A_inv * B

		buf_2.load("D_inv.bin");
		gf_matrix_mul<<<dim3(16,16), dim3(BLOCK_DIM, BLOCK_DIM)>>>
			(buf_3.toKernel(), buf_2.toKernel(), buf_1.toKernel());
		cudaDeviceSynchronize();
		// buf_1 = A_inv * B * D_inv
		// buf_2 = D_inv
		// buf_3 = A_inv * B

		buf_1.write_host();
		buf_2.write_host();
		buf_3.load("A_inv.bin");

		for (uint32_t i = 0; i < M; ++i) {
			for (uint32_t j = 0; j < M; ++j) {
				std::cout << std::hex << buf_3.toHost()->data[i][j] << ' ';
			}
			for (uint32_t j = 0; j < M; ++j) {
				std::cout << std::hex << buf_1.toHost()->data[i][j] << ' ';
			}
			std::cout << std::endl;
		}

		for (uint32_t i = 0; i < M; ++i) {
			for (uint32_t j = 0; j < M; ++j) {
				std::cout << std::hex << gf_int_t(0) << ' ';
			}
			for (uint32_t j = 0; j < M; ++j) {
				std::cout << std::hex << buf_2.toHost()->data[i][j] << ' ';
			}
			std::cout << std::endl;
		}
	
	} catch (std::exception &e) {
		std::cout << "Error: " << e.what() << std::endl;
		throw e;
	}

	return 0;
}