#include <exception>

#include <utility>

#include <string>

#include <fstream>
#include <ios>



/* ************************************************** */
/* Customized Exception Types */

struct bad_cuda_alloc: public std::exception
{
	const char * what() const noexcept override {
		return "Failed to allocate enough managed memory.";
	}
};

struct bad_managed_deref: public std::exception
{
	const char * what() const noexcept override {
		return "Connot dereference a null managed_ptr.";
	}
};

struct bad_file_stream: public std::exception
{
	const char * what() const noexcept override {
		return "Failed to open a file.";
	}
};



/* ************************************************** */
/* Smart Pointer to CUDA managed memory */

template <typename T> class managed_ptr;

template <typename T>
managed_ptr<T> make_managed(int size = 1);



template <typename T>
class managed_ptr {
	
	friend managed_ptr make_managed<T>(int size);

protected:

	T *raw;
	int *_count;
	// A pointer to a piece of managed memory,
	// which automatically migrates between host and device,
	// and its reference-counting.

	explicit managed_ptr(T *_raw): raw(_raw), _count(new int(1)) {}

	void try_release() {
		if (--(*_count) == 0) {	
			cudaFree(raw);
			delete _count;
		}
	}

public:

	managed_ptr(const managed_ptr &another): raw(another.raw), _count(another._count) {
		++(*_count);
	}
	managed_ptr & operator=(const managed_ptr &another) {
		try_release();
		
		raw = another.raw;
		_count = another._count;
		++(*_count);

		return *this;
	}
	managed_ptr(managed_ptr &&copiee): raw(copiee.raw), _count(copiee._count) {}
	managed_ptr & operator=(managed_ptr &&copiee) {
		try_release();

		raw = copiee.raw;
		_count = copiee._count;

		return *this;		
	}
	~managed_ptr() {
		try_release();
	}

	operator T * () const {
		return raw;
	}

	T & operator*() const {
		return *raw;
	}

	T * operator->() const {
		return raw;
	}
};



template <typename T>
managed_ptr<T> make_managed(int size)
{
	T *raw;

	cudaError_t flag = cudaMallocManaged(&raw, sizeof(T) * size);
	if (flag != cudaSuccess) {
		throw bad_cuda_alloc();
	}

	return managed_ptr<T>(raw);
}



/* ************************************************** */
/* Loads data from and stores it to external memory */

template <typename T>
class cuder: public managed_ptr<T>
{

public:

	cuder(const managed_ptr<T> &&copiee): managed_ptr<T>(copiee) {}

	void load(void init_fun(T *)) {
		init_fun(static_cast<T *>(*this));
	}

	void store(const std::string &file_name) const {
		std::ofstream ofs("tmp/" + file_name + ".bin", std::ios::binary);
		if (!ofs)
			throw bad_file_stream();

		long long size = sizeof(T) / sizeof(char);
		ofs.write(reinterpret_cast<const char *>(static_cast<T *>(*this)), size);

		ofs.close();
	}

	void load(const std::string &file_name) {
		std::ifstream ifs("tmp/" + file_name + ".bin", std::ios::binary);
		if (!ifs)
			throw bad_file_stream();
		
		long long size = sizeof(T) / sizeof(char);
		ifs.read(reinterpret_cast<char *>(static_cast<T *>(*this)), size);

		ifs.close();
	}

};



template <typename T>
cuder<T> make_cuder(int size = 1)
{
	return cuder<T>(std::move(make_managed<T>(size)));
}