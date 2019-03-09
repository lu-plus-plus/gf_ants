
#include "gf_int.h"
#include <iostream>



using gf_test_t = gf_int<4>;

__global__ void test(gf_test_t *_a, gf_test_t *_b)
{
	auto &a = *_a;
	auto &b = *_b;
	a = gf_test_t(14);
	b = gf_test_t(11);

	auto r1 = a+b;
	auto r2 = clmul_least(a, b);
    auto r3 = clmul_most(a, b);
    auto r4 = gf_mul(a, b);

	if (threadIdx.z == 0) {
		printf("0x%llx\n", static_cast<uint64_t>(r1.value()));
		printf("0x%llx\n", static_cast<uint64_t>(r2.value()));
        printf("0x%llx\n", static_cast<uint64_t>(r3.value()));
        printf("0x%llx\n", static_cast<uint64_t>(gf_test_t::irred_g));
        printf("0x%llx\n", static_cast<uint64_t>(gf_test_t::g_star));
        printf("0x%llx\n", static_cast<uint64_t>(gf_test_t::q_plus));
        printf("0x%llx\n", static_cast<uint64_t>(r4.value()));
	}

}

int main(void)
{
	gf_test_t *a, *b;
	cudaMalloc(&a, sizeof(gf_test_t));
	cudaMalloc(&b, sizeof(gf_test_t));

	test<<<dim3(1, 1, 1), dim3(1, 1, 8)>>>(a, b);

	cudaFree(a);
	cudaFree(b);

	return 0;
}