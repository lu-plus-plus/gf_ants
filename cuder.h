#include <exception>
#include <string>
#include <fstream>
#include <ios>



struct bad_cuda_alloc: public std::exception
{
	const char * what() const noexcept override {
		return "Failed to allocate enough memory on GPU.";
	}
};

struct bad_remote_deref: public std::exception
{
	const char * what() const noexcept override {
		return "Dereference a null remote-ptr.";
	}
};

struct bad_file_stream: public std::exception
{
	const char * what() const noexcept override {
		return "Failed to open a file.";
	}
};



template <typename T>
class remote_ptr {
private:
	T *raw;
	int *_count;
	// A pointer to T[0...(n-1)], and its reference counting.

	void try_release() {
		if (--(*_count) == 0) {	
			cudaFree(raw);
			raw = nullptr;

			delete _count;
			_count = nullptr;
		}
	}

public:
	remote_ptr(T *_raw): raw(_raw), _count(new int(1)) {}
	remote_ptr(const remote_ptr &another): raw(another.raw), _count(another._count) {
		++(*_count);
	}
	remote_ptr & operator=(const remote_ptr &another) {
		try_release();
		
		raw = another.raw;
		_count = another._count;
		++(*_count);

		return *this;
	}
	~remote_ptr() {
		try_release();
	}

	T * toKernel() {
		if (raw == nullptr)
			throw bad_remote_deref();
		
		return raw;
	}
};

template <typename T>
remote_ptr<T> make_remote(int length = 1) {
	T *raw;
	cudaError_t flag = cudaMalloc(&raw, sizeof(T) * length);
	if (flag != cudaSuccess) {
		throw bad_cuda_alloc();
	}
	return remote_ptr<T>(raw);
}



template <typename T>
class cuder
{
private:
	T *host_ptr;
	remote_ptr<T> dev_ptr;

public:
	cuder(T *_host, const remote_ptr<T> &_dev): host_ptr(_host), dev_ptr(_dev) {}

	void init(void init_fun(T *)) {
		init_fun(host_ptr);
		cudaMemcpy(dev_ptr.toKernel(), host_ptr, sizeof(T), cudaMemcpyHostToDevice);
	}

	void write_host() {
		cudaMemcpy(host_ptr, dev_ptr.toKernel(), sizeof(T), cudaMemcpyDeviceToHost);
	}

	void write_disk(const std::string &file_name) {
		std::ofstream ofs("temp/" + file_name, std::ios::binary);
		if (!ofs)
			throw bad_file_stream();
		
		write_host();

		long long size = sizeof(T) / sizeof(char);
		ofs.write(reinterpret_cast<const char *>(host_ptr), size);

		ofs.close();
	}

	void load(const std::string &file_name) {
		std::ifstream ifs("temp/" + file_name, std::ios::binary);
		if (!ifs)
			throw bad_file_stream();
		
		long long size = sizeof(T) / sizeof(char);
		ifs.read(reinterpret_cast<char *>(host_ptr), size);

		cudaMemcpy(dev_ptr.toKernel(), host_ptr, sizeof(T), cudaMemcpyHostToDevice);

		ifs.close();
	}

	T * toHost() {
		return host_ptr;
	}
	
	T * toKernel() {
		return dev_ptr.toKernel();
	}
};