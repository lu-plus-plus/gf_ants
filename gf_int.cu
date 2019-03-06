
#include "gf_int.h"
#include <iostream>



__global__ void test(gf_int8_t *_a, gf_int8_t *_b)
{
	auto &a = *_a;
	auto &b = *_b;
	a = 14;
	b = 11;

	auto r1 = a+b;
	auto r2 = clmul_least(a, b);
	auto r3 = clmul_most(a, b);

	if (threadIdx.z == 0) {
		printf("%llu\n", static_cast<uint64_t>(r1.getData()));
		printf("%llu\n", static_cast<uint64_t>(r2.getData()));
		printf("%llu\n", static_cast<uint64_t>(r3.getData()));
	}

}

int main(void)
{
	gf_int8_t *a, *b;
	cudaMalloc(&a, sizeof(gf_int8_t));
	cudaMalloc(&b, sizeof(gf_int8_t));

	test<<<dim3(1, 1, 1), dim3(1, 1, 8)>>>(a, b);

	cudaFree(a);
	cudaFree(b);

	return 0;
}