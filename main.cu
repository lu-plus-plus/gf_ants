
#include "gf_int.h"
#include <iostream>



template <typename T>
T * make_cuder(int length = 1) {
	T *ptr;
	cudaMalloc(&ptr, sizeof(T)*length);
	return ptr;
}

template <typename T>
class cuder {
private:
	T *ptr;
	int *_count;
	// A pointer to T[0...(n-1)], and its reference counting
	// Since the memory is allocated in GPU but cuder is on host,
	// no constructor/decons need to be invoked.

	void try_release() {
		if (--(*_count) == 0)
			cudaFree(ptr);
	}

public:
	cuder(T *raw_ptr): ptr(raw_ptr), _count(new int(1)) {}
	cuder(const cuder &another): ptr(another.ptr), _count(another._count) {
		++(*_count);
	}
	cuder & operator=(const cuder &another) {
		try_release();
		
		ptr = another.ptr;
		_count = another._count;
		++(*_count);
	}
	~cuder() {
		try_release();
	}

	T * toKernel() {
		return ptr;
	}
};


using gf_test_t = gf_int<8>;

__global__ void test(gf_test_t *_a, gf_test_t *_b)
{
	auto &a = *_a;
	auto &b = *_b;
	a = gf_test_t(0xc5);
	b = gf_test_t(0xd4);

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
	cuder<gf_test_t> a(make_cuder<gf_test_t>());
	cuder<gf_test_t> b(make_cuder<gf_test_t>());

	test<<<dim3(1, 1, 1), dim3(1, 1, 8)>>>(a.toKernel(), b.toKernel());

	return 0;
}