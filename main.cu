
#include "gf_matrix.h"
#include "cuder.h"

// constexpr bool PRINT_VERBOSE = 0;
// constexpr bool PRINT_INITIAL_VALUE = 0;
constexpr bool PRINT_RESULT = true;



constexpr int TOTAL_DIM = 2048;

constexpr int CAPABLE_DIM = TOTAL_DIM / 2;

constexpr int BITS = 8;

using gf_int_t = gf_int<BITS>;
using square_t = gf_square<gf_int_t, CAPABLE_DIM>;



void square_inverse(square_t *d_A_ptr, square_t *d_B_ptr)
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

void square_mul(square_t *d_A_ptr, square_t *d_B_ptr, square_t *d_C_ptr)
{
	dim3 grid(16, 16);
	dim3 block(BLOCK_DIM, BLOCK_DIM);

	gf_matrix_mul<<<grid, block>>>(d_A_ptr, d_B_ptr, d_C_ptr);
	cudaDeviceSynchronize();
}

void square_add(square_t *d_A_ptr, square_t *d_B_ptr)
{
	dim3 grid(16, 16);
	dim3 block(BLOCK_DIM, BLOCK_DIM);

	gf_matrix_add<<<grid, block>>>(d_A_ptr, d_B_ptr);
	cudaDeviceSynchronize();
}



void init_A(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = 0; j < i; ++j) {
			obj->data[i][j] = gf_int_t(0);
		}
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

void init_C(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
			obj->data[i][j] = gf_int_t(0);
		}
	}
}

void init_D(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = 0; j < i; ++j) {
			obj->data[i][j] = gf_int_t(0);
		}
		for (uint32_t j = i; j < CAPABLE_DIM; ++j) {
			obj->data[i][j] = gf_int_t(2*CAPABLE_DIM + i + j + 1);
		}
	}
}

void init_identify(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
			obj->data[i][j] = gf_int_t(0);
		}
		
		obj->data[i][i] = gf_int_t(1);
	}
}



int main(void)
{
	if (CAPABLE_DIM % BLOCK_DIM != 0) {
		std::cout << "The size of square_t isn't multiple of block size." << std::endl;
		throw std::exception();
	}
	if (TOTAL_DIM != CAPABLE_DIM * 2) {
		std::cout << "The size of square matrix must be twice capable_dim." << std::endl;
		throw std::exception();
	}

	try {
		cuder<square_t> buf_1(make_cuder<square_t>());
		cuder<square_t> buf_2(make_cuder<square_t>());
		cuder<square_t> buf_3(make_cuder<square_t>());
		cuder<square_t> buf_4(make_cuder<square_t>());

		buf_1.load(init_A);
		buf_2.load(init_identify);
		square_inverse(buf_1, buf_2);
		// buf_2 = A_inv

		buf_1.load(init_C);
		square_mul(buf_1, buf_2, buf_3);
		// buf_2 = A_inv
		// buf_3 = C * A_inv

		buf_1.load(init_B);
		square_mul(buf_3, buf_1, buf_4);
		// buf_2 = A_inv
		// buf_3 = C * A_inv
		// buf_4 = C * A_inv * B

		buf_1.load(init_D);
		square_add(buf_4, buf_1);
		// buf_2 = A_inv
		// buf_3 = C * A_inv
		// buf_4 = D - C * A_inv * B

		buf_1.load(init_identify);
		square_inverse(buf_4, buf_1);
		// buf_1 = DR
		// buf_2 = A_inv
		// buf_3 = C * A_inv

		square_mul(buf_1, buf_3, buf_4);
		// buf_1 = DR
		// buf_2 = A_inv
		// buf_3 = C * A_inv
		// buf_4 = DL

		buf_3.store("C_Ainv");
		buf_4.store("DL");
		// buf_1 = DR
		// buf_2 = A_inv
		
		buf_3.load(init_B);
		square_mul(buf_2, buf_3, buf_4);
		// buf_1 = DR
		// buf_2 = A_inv
		// buf_4 = A_inv * B

		square_mul(buf_4, buf_1, buf_3);
		// buf_1 = DR
		// buf_2 = A_inv
		// buf_3 = UR
		// buf_4 = A_inv * B

		buf_1.store("DR");
		buf_4.load("C_Ainv");
		// buf_2 = A_inv
		// buf_3 = UR
		// buf_4 = C * A_inv

		square_mul(buf_3, buf_4, buf_1);
		// buf_1 = UR * C * A_inv
		// buf_2 = A_inv
		// buf_3 = UR
		// buf_4 = C * A_inv

		square_add(buf_1, buf_2);
		// buf_1 = UL
		// buf_2 = A_inv
		// buf_3 = UR
		// buf_4 = C * A_inv

		buf_2.load("DL");
		buf_4.load("DR");
		// buf_1 = Up Left
		// buf_2 = Down Left
		// buf_3 = Up Right
		// buf_4 = Down Right

		if (PRINT_RESULT) {
			for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
				for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
					std::cout << std::hex << buf_1->data[i][j] << ' ';
				}
				for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
					std::cout << std::hex << buf_3->data[i][j] << ' ';
				}
				std::cout << std::endl;
			}
	
			for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
				for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
					std::cout << std::hex << buf_2->data[i][j] << ' ';
				}
				for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
					std::cout << std::hex << buf_4->data[i][j] << ' ';
				}
				std::cout << std::endl;
			}
		}	
	
	} catch (std::exception &e) {
		std::cout << "Error: " << e.what() << std::endl;
		throw e;
	}

	return 0;
}