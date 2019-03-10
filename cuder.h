
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