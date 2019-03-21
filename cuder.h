#include <exception>



template <typename T>
T * make_cuder(int length = 1) {
	T *ptr;
	cudaError_t flag = cudaMalloc(&ptr, sizeof(T) * length);
	if (flag != cudaSuccess) {
		throw std::bad_alloc();
	}
	return ptr;
}

template <typename T>
class cuder {
private:
	T *ptr;
	int *_count;
	// A pointer to T[0...(n-1)], and its reference counting.

	void try_release() {
		if (--(*_count) == 0) {
			cudaFree(ptr);
			delete _count;
		}
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