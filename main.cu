
#include "gf_matrix.h"
#include "cuder.h"

constexpr bool PRINT_VERBOSE = 1;
constexpr bool PRINT_INITIAL_VALUE = 0;
constexpr bool PRINT_RESULT = 0;



constexpr int M = 20480;

constexpr int BITS = 8;

using gf_int_t = gf_int<BITS>;
using square_t = gf_square<gf_int_t, M>;



// Allocate static memory on host
square_t h_A;
square_t h_B;

int main(void)
{
	if (M % BLOCK_DIM != 0) {
		std::cout << "This square matrix isn't block-size-aligned." << std::endl;
		throw std::exception();
	}

	try {
		// Allocate heap memory on device
		cuder<square_t> d_A_ptr(make_cuder<square_t>());
		cuder<square_t> d_B_ptr(make_cuder<square_t>());

		// Initialize host data and/or print it
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
		
		// Initialize device data
		cudaMemcpy(d_A_ptr.toKernel(), &h_A, sizeof(h_A), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B_ptr.toKernel(), &h_B, sizeof(h_B), cudaMemcpyHostToDevice);

		// Time counter
		cudaEvent_t start, stop;
		if (PRINT_VERBOSE) {
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);
		}



		// Guassian-Jordan Elimination

		dim3 grid(GRID_DIM_X, 1);
		dim3 block(BLOCK_DIM, BLOCK_DIM);
		
		for (int pivot_block_idx = 0; pivot_block_idx < (M / BLOCK_DIM); ++pivot_block_idx) {
			
			elimination_round<<<grid, block>>>(
				d_A_ptr.toKernel(), d_B_ptr.toKernel(), pivot_block_idx);
			cudaDeviceSynchronize();

			if (PRINT_VERBOSE) {
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				std::cout << "Round " << pivot_block_idx << ": "
					<< (elapsedTime / 1000) << " s"
					<< std::endl;
			}	
		
		}

		normalize_by_pivots<<<grid, block>>>(d_A_ptr.toKernel(), d_B_ptr.toKernel());
		cudaDeviceSynchronize();
		


		// Copy the result back to host and/or print it
		cudaMemcpy(&h_B, d_B_ptr.toKernel(), sizeof(h_B), cudaMemcpyDeviceToHost);

		if (PRINT_RESULT) {
			for (uint32_t i = 0; i < M; ++i) {
				for (uint32_t j = 0; j < M; ++j) {
					std::cout << std::hex << h_B.data[i][j] << ' ';
				}
				std::cout << std::endl;
			}
		}
	
	} catch (std::bad_alloc &e) {
		std::cout << "Failed to allocate enough memory on GPU." << std::endl;
		throw e;
	}

	return 0;
}