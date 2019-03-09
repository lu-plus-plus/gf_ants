
#include "gf_int.h"
#include <iostream>



using gf_int5_t = gf_int<5>;

__global__ void test(gf_int5_t *_a, gf_int5_t *_b)
{
	auto &a = *_a;
	auto &b = *_b;
	a = gf_int5_t(14);
	b = gf_int5_t(11);

	auto r1 = a+b;
	auto r2 = clmul_least(a, b);
	auto r3 = clmul_most(a, b);

	if (threadIdx.z == 0) {
		printf("0x%llx\n", static_cast<uint64_t>(r1.value()));
		printf("0x%llx\n", static_cast<uint64_t>(r2.value()));
		printf("0x%llx\n", static_cast<uint64_t>(r3.value()));
	}

}

int main(void)
{
	gf_int5_t *a, *b;
	cudaMalloc(&a, sizeof(gf_int5_t));
	cudaMalloc(&b, sizeof(gf_int5_t));

	test<<<dim3(1, 1, 1), dim3(1, 1, 8)>>>(a, b);

	cudaFree(a);
	cudaFree(b);

	return 0;
}