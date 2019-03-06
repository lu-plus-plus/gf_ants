
#include <cstdint>
#include <type_traits>



template <bool IF>
using bool_constant = std::integral_constant<bool, IF>;

constexpr bool in_closed_range(const int val, const int l, const int r)
{
	return l <= val && val <= r;
}

template <int BITS, typename = bool_constant<true>>
struct gf_underlying;

template <int BITS>
struct gf_underlying<BITS, bool_constant<in_closed_range(BITS, 1, 8)>> {
	using type = uint8_t;
};

template <int BITS>
struct gf_underlying<BITS, bool_constant<in_closed_range(BITS, 9, 16)>> {
	using type = uint16_t;
};

template <int BITS>
struct gf_underlying<BITS, bool_constant<in_closed_range(BITS, 17, 32)>> {
	using type = uint32_t;
};

template <int BITS>
struct gf_underlying<BITS, bool_constant<in_closed_range(BITS, 33, 64)>> {
	using type = uint64_t;
};



template <int BITS>
class gf_int;



template <int BITS>
__device__ gf_int<BITS> operator+(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);

template <int BITS>
__device__ gf_int<BITS> clmul_least(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);

template <int BITS>
__device__ gf_int<BITS> clmul_most(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);




template <int BITS>
class gf_int
{
public:
	using underlying_t = typename gf_underlying<BITS>::type;

private:
	underlying_t data;

public:
	gf_int() = default;
	__device__ gf_int(underlying_t raw_data): data(raw_data) {}
	gf_int(const gf_int &) = default;
	gf_int & operator=(const gf_int &rhs) = default;
	~gf_int() = default;

	__device__ underlying_t getData() { return data; }

	__device__ gf_int & operator+=(const gf_int &rhs) {
		data ^= rhs.data;
		return *this;
	}

	friend __device__ gf_int operator+<BITS>(const gf_int &lhs, const gf_int &rhs);

	__device__ gf_int & clmuled_least_by(const gf_int &rhs) {
		gf_int &lhs = *this;
	
		int digit = threadIdx.z;
		__shared__ underlying_t c[BITS];

		c[digit] = lhs.data * (rhs.data & (1 << digit));
		__syncthreads();

		for (int stride = 2; stride <= BITS; stride *= 2) {
			if (digit % stride == 0) {
				c[digit] ^= c[digit + (stride >> 1)];
			}
			__syncthreads();
		}

		lhs.data = c[0];
		return lhs;
	}

	__device__ gf_int & clmuled_most_by(const gf_int &rhs) {
		gf_int &lhs = *this;
	
		int digit = threadIdx.z;
		__shared__ underlying_t c[BITS];

		c[digit] = (lhs.data >> (BITS - digit)) * ((rhs.data >> digit) & 1);
		__syncthreads();

		for (int stride = 2; stride <= BITS; stride *= 2) {
			if (digit % stride == 0) {
				c[digit] ^= c[digit + (stride >> 1)];
			}
			__syncthreads();
		}

		lhs.data = c[0];
		return lhs;
	}

	friend __device__ gf_int clmul_least<BITS>(const gf_int &lhs, const gf_int &rhs);
	friend __device__ gf_int clmul_most<BITS>(const gf_int &lhs, const gf_int &rhs);
};

template <int BITS>
__device__ gf_int<BITS> operator+(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs)
{
	gf_int<BITS> result(lhs);
	result += rhs;
	return result;
}

template <int BITS>
__device__ gf_int<BITS> clmul_least(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs)
{
	gf_int<BITS> result(lhs);
	result.clmuled_least_by(rhs);
	return result;
}

template <int BITS>
__device__ gf_int<BITS> clmul_most(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs)
{
	gf_int<BITS> result(lhs);
	result.clmuled_most_by(rhs);
	return result;
}



using gf_int8_t = gf_int<8>;