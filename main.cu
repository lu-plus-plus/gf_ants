
#include "gf_matrix.h"
#include "cuder.h"

// constexpr bool PRINT_VERBOSE = 0;
// constexpr bool PRINT_INITIAL_VALUE = 0;
// constexpr bool PRINT_RESULT = 0;



constexpr int TOTAL_DIM = 2048;

constexpr int CAPABLE_DIM = 1024;

constexpr int BITS = 8;

using gf_int_t = gf_int<BITS>;
using square_t = gf_square<gf_int_t, CAPABLE_DIM>;



void gaussian_elimination(square_t *d_A_ptr, square_t *d_B_ptr)
{
	dim3 grid(GRID_DIM_X);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	
	const int block_level_dim = CAPABLE_DIM / BLOCK_DIM;

	for (int pivot_block_idx = 0; pivot_block_idx < block_level_dim; ++pivot_block_idx) {
		elimination_round<<<grid, block>>>(d_A_ptr, d_B_ptr, pivot_block_idx);
		cudaDeviceSynchronize();
	}

	normalize_by_pivots<<<grid, block>>>(d_A_ptr, d_B_ptr);
	cudaDeviceSynchronize();
}



void init_A(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = i; j < CAPABLE_DIM; ++j) {
			obj->data[i][j] = gf_int_t(i + j + 1);
		}
	}
}

void init_B(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
			obj->data[i][j] = gf_int_t(i + (CAPABLE_DIM + j) + 1);
		}
	}
}

void init_D(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = i; j < CAPABLE_DIM; ++j) {
			obj->data[i][j] = gf_int_t(2*CAPABLE_DIM + i + j + 1);
		}
	}
}

void host_identify(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		obj->data[i][i] = gf_int_t(1);
	}
}



int main(void)
{
	if (CAPABLE_DIM % BLOCK_DIM != 0) {
		std::cout << "The size of square_t isn't multiple of block size." << std::endl;
		throw std::exception();
	}
	if (TOTAL_DIM % CAPABLE_DIM != 0) {
		std::cout << "The size of square matrix isn't multiple of capable_dim." << std::endl;
		throw std::exception();
	}

	try {
		cuder<square_t> buf_1(make_cuder<square_t>());
		cuder<square_t> buf_2(make_cuder<square_t>());
		cuder<square_t> buf_3(make_cuder<square_t>());

		buf_1.load(init_A);
		buf_2.load(host_identify);
		gaussian_elimination(buf_1, buf_2);
		buf_2.store("A_inv.bin");
		// buf_1 = undefined
		// buf_2 = A_inv
		// buf_3 = undefined

		buf_1.load(init_D);
		buf_3.load(host_identify);
		gaussian_elimination(buf_1, buf_3);
		buf_3.store("D_inv.bin");
		// buf_1 = undefined
		// buf_2 = A_inv
		// buf_3 = D_inv

		buf_1.load(init_B);
		gf_matrix_mul<<<dim3(16,16), dim3(BLOCK_DIM, BLOCK_DIM)>>>(
			buf_2.c_ptr(), buf_1.c_ptr(), buf_3.c_ptr());
		cudaDeviceSynchronize();
		// buf_1 = B
		// buf_2 = A_inv
		// buf_3 = A_inv * B

		buf_2.load("D_inv.bin");
		gf_matrix_mul<<<dim3(16,16), dim3(BLOCK_DIM, BLOCK_DIM)>>>(
			buf_3.c_ptr(), buf_2.c_ptr(), buf_1.c_ptr());
		cudaDeviceSynchronize();
		// buf_1 = A_inv * B * D_inv
		// buf_2 = D_inv
		// buf_3 = A_inv * B

		buf_3.load("A_inv.bin");

		for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
			for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
				std::cout << std::hex << buf_3->data[i][j] << ' ';
			}
			for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
				std::cout << std::hex << buf_1->data[i][j] << ' ';
			}
			std::cout << std::endl;
		}

		for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
			for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
				std::cout << std::hex << gf_int_t(0) << ' ';
			}
			for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
				std::cout << std::hex << buf_2->data[i][j] << ' ';
			}
			std::cout << std::endl;
		}
	
	} catch (std::exception &e) {
		std::cout << "Error: " << e.what() << std::endl;
		throw e;
	}

	return 0;
}