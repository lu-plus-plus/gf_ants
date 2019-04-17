
#include "gf_matrix.h"
#include "cuder.h"

#define APP_BITS (8)

#define APP_DIM (2048)

#define CAPABLE_DIM (512)

constexpr bool PRINT_TOTAL_TIME = true;
constexpr bool PRINT_VERBOSE = true;
constexpr bool PRINT_RESULT = false;



using gf_int_t = gf_int<APP_BITS>;

using capable_t = gf_square<gf_int_t, CAPABLE_DIM>;



void capable_inverse(capable_t *d_A_ptr, capable_t *d_B_ptr)
{
	dim3 grid(INVERSE_GRID_DIM_X);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	
	const int block_level_dim = CAPABLE_DIM / BLOCK_DIM;

	for (int pivot_block_idx = 0; pivot_block_idx < block_level_dim; ++pivot_block_idx) {
		elimination_round<<<grid, block>>>(d_A_ptr, d_B_ptr, pivot_block_idx);
		cudaDeviceSynchronize();
	}

	normalize_by_pivots<<<grid, block>>>(d_A_ptr, d_B_ptr);
	cudaDeviceSynchronize();
}

void capable_mul(capable_t *d_A_ptr, capable_t *d_B_ptr, capable_t *d_C_ptr)
{
	dim3 grid(16, 16);
	dim3 block(BLOCK_DIM, BLOCK_DIM);

	gf_matrix_mul<<<grid, block>>>(d_A_ptr, d_B_ptr, d_C_ptr);
	cudaDeviceSynchronize();
}

void capable_add(capable_t *d_A_ptr, capable_t *d_B_ptr)
{
	dim3 grid(16, 16);
	dim3 block(BLOCK_DIM, BLOCK_DIM);

	gf_matrix_add<<<grid, block>>>(d_A_ptr, d_B_ptr);
	cudaDeviceSynchronize();
}



void init_identify(capable_t *obj) {
	for (uint32_t i = 0; i < CAPABLE_DIM; ++i) {
		for (uint32_t j = 0; j < CAPABLE_DIM; ++j) {
			obj->data[i][j] = gf_int_t(0);
		}
		obj->data[i][i] = gf_int_t(1);
	}
}



inline std::string piece(int i, int j) {
	return std::to_string(APP_BITS) + "_"
		+ std::to_string(APP_DIM) + "_"
		+ std::to_string(i) + "_" + std::to_string(j);
}

int main(void)
{
	if (CAPABLE_DIM % BLOCK_DIM != 0) {
		std::cout << "The capable size must be multiple of the block size." << std::endl;
		throw std::exception();
	}
	if (APP_DIM % CAPABLE_DIM != 0) {
		std::cout << "The total size must be multiple of the capable size." << std::endl;
		throw std::exception();
	}

	try {
		cuder<capable_t> buf_1(make_cuder<capable_t>());
		cuder<capable_t> buf_2(make_cuder<capable_t>());
		cuder<capable_t> buf_3(make_cuder<capable_t>());
		cuder<capable_t> buf_4(make_cuder<capable_t>());

		// Time Counter Initialization
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// Initialize
		constexpr int pieceDim = APP_DIM / CAPABLE_DIM;

		for (int row = 0; row < pieceDim; ++row) {
			for (int col = 0; col < pieceDim; ++col) {
				capable_t &buf = *buf_1;

				int glb_i = row * CAPABLE_DIM;
				for (int i = 0; i < CAPABLE_DIM; ++i) {

					int glb_j = col * CAPABLE_DIM;
					for (int j = 0; j < CAPABLE_DIM; ++j) {
						buf.data[i][j] = (glb_i <= glb_j) ? gf_int_t(glb_i + glb_j + 1) : gf_int_t(0);
						++glb_j;
					}
		
					++glb_i;
				}

				buf_1.store(piece(row, col));
			}

			for (int col = 0; col < pieceDim; ++col) {
				capable_t &buf = *buf_1;

				int glb_i = row * CAPABLE_DIM;
				for (int i = 0; i < CAPABLE_DIM; ++i) {

					int glb_j = col * CAPABLE_DIM;
					for (int j = 0; j < CAPABLE_DIM; ++j) {
						buf.data[i][j] = (glb_i == glb_j) ? gf_int_t(1) : gf_int_t(0);
						++glb_j;
					}
		
					++glb_i;
				}

				buf_1.store(piece(row, pieceDim + col));
			}
		}

		cudaEventRecord(start, 0);

		// Calculate
		for (int pivot_idx = 0; pivot_idx < pieceDim; ++pivot_idx) {
			cudaEvent_t round_start, round_stop;
			if (PRINT_VERBOSE) {
				cudaEventCreate(&round_start);
				cudaEventCreate(&round_stop);
				cudaEventRecord(round_start, 0);
			}

			buf_2.load(piece(pivot_idx, pivot_idx));
			buf_1.load(init_identify);
			capable_inverse(buf_2, buf_1);
			// buf_1 = inv(pivot_idx,pivot_idx)

			for (int row = 0; row < pieceDim; ++row) {
				if (row != pivot_idx) {
					buf_2.load(piece(row, pivot_idx));
					capable_mul(buf_2, buf_1, buf_3);
					// buf_3 = coeff = (row, pivot) * (pivot, pivot)_inv
	
					for (int col = pivot_idx; col <= pivot_idx + pieceDim; ++col) {
						buf_2.load(piece(pivot_idx, col));
						capable_mul(buf_3, buf_2, buf_4);

						buf_2.load(piece(row, col));
						capable_add(buf_2, buf_4);
						buf_2.store(piece(row, col));
					}
				}
			}

			for (int col = pivot_idx; col <= pivot_idx + pieceDim; ++col) {
				if (col != pivot_idx) {
					buf_2.load(piece(pivot_idx, col));
					capable_mul(buf_1, buf_2, buf_3);
					buf_3.store(piece(pivot_idx, col));
				} else {
					buf_3.load(init_identify);
					buf_3.store(piece(pivot_idx, col));
				}
			}

			if (PRINT_VERBOSE) {
				cudaEventRecord(round_stop, 0);
				cudaEventSynchronize(round_stop);

				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, round_start, round_stop);
				std::cout << "Round " << pivot_idx << ": "
					<< (elapsedTime/1000) << " s."
					<< std::endl;
			}
		}
		
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		if (PRINT_RESULT) {
			for (int i = 0; i < pieceDim; ++i) {
				std::ifstream ifs[pieceDim];
				gf_int_t buf[CAPABLE_DIM];

				for (int j = 0; j < pieceDim; ++j) {
					ifs[j].open("tmp/" + piece(i, pieceDim+j) + ".bin", std::ios::binary);
					if (!ifs[j])
						throw bad_file_stream();
				}

				long long size = sizeof(gf_int_t) * CAPABLE_DIM / sizeof(char);
				for (int row = 0; row < CAPABLE_DIM; ++row) {
					for (int col = 0; col < pieceDim; ++col) {
						ifs[col].read(reinterpret_cast<char *>(buf), size);
						
						for (int k = 0; k < CAPABLE_DIM; ++k) {
							std::cout << std::hex << buf[k] << ' ';
						}
					}
					std::cout << std::endl;
				}

				for (int j = 0; j < pieceDim; ++j) {
					ifs[j].close();
				}
			}
		}

		if (PRINT_TOTAL_TIME) {
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime, start, stop);
			std::cout << "Total Time: " << (elapsedTime/1000) << " s." << std::endl;
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