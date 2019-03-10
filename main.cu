
#include "gf_int.h"
#include "cuder.h"

constexpr int BITS = 8;
constexpr int N = 256;
using gf_test_t = gf_int<BITS>;



__global__ void CalcuInverse(gf_test_t *host_matrix)
{
	for (uint32_t i = 1; i < 256; ++i) {
		for (uint32_t j = 1; j < 256; ++j) {
			if ( (gf_test_t(i) * gf_test_t(j)).value() == gf_test_t(1).value() ) {
				host_matrix[i] = gf_test_t(j);
				break;
			}
		}
	}
}

gf_test_t host_matrix[N];

int main(void)
{
	cuder<gf_test_t> device_matrix(make_cuder<gf_test_t>(N));
	cudaMemcpy(device_matrix.toKernel(), host_matrix, sizeof(host_matrix), cudaMemcpyHostToDevice);

	CalcuInverse<<<dim3(1,1,1), dim3(1,1,BITS)>>>(device_matrix.toKernel());

	cudaMemcpy(host_matrix, device_matrix.toKernel(), sizeof(host_matrix), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			std::cout << "0x" << std::hex << host_matrix[i * 16 + j] << ' ';
		}
		std::cout << std::endl;
	}

	return 0;
}