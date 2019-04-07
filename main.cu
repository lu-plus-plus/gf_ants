
#include "gf_matrix.h"
#include "cuder.h"

#define APP_BITS (8)

#define APP_DIM (2048)

constexpr bool PRINT_TIME = true;
constexpr bool PRINT_RESULT = false;



using gf_int_t = gf_int<APP_BITS>;

constexpr int CAPABLE_DIM = APP_DIM;
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

void init_whole(square_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = 0; j < i; ++j) {
			obj->data[i][j] = gf_int_t(0);
		}
		for (uint32_t j = i; j < CAPABLE_DIM; ++j) {
			obj->data[i][j] = gf_int_t(i + j + 1);
		}
	}
}



int main(void)
{
	if (CAPABLE_DIM % BLOCK_DIM != 0) {
		std::cout << "The size of square_t isn't multiple of block size." << std::endl;
		throw std::exception();
	}

	try {
		cuder<square_t> buf_1(make_cuder<square_t>());
		cuder<square_t> buf_2(make_cuder<square_t>());

		// Time Counter Initialization
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		if (PRINT_TIME) {
			cudaEventRecord(start, 0);
		}

		buf_1.load(init_whole);
		buf_2.load(init_identify);
		square_inverse(buf_1, buf_2);

		if (PRINT_TIME) {
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime, start, stop);
			std::cout << "Total Time: " << elapsedTime / 1000 << " ms." << std::endl;
		}

		if (PRINT_RESULT) {
			for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
				for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
					std::cout << std::hex << buf_2->data[i][j] << ' ';
				}
				std::cout << std::endl;
			}
		}
	
		// Time Counter Destruction
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

	} catch (std::exception &e) {
		std::cout << "Error: " << e.what() << std::endl;
		throw e;
	}

	return 0;
}