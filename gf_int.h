#include <cstdint>
#include <type_traits>



// Type Traits in Compile-time
// ****************************************



template <bool IF>
using bool_constant = std::integral_constant<bool, IF>;



// underlying int type inside class gf_int

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

// ****************************************





// Class gf_int
// ****************************************



// declaration of
// gf_int and its friend functions

template <int BITS>
class gf_int;

template <int BITS>
__device__ gf_int<BITS> operator+(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);

template <int BITS>
__device__ gf_int<BITS> clmul_least(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);

template <int BITS>
__device__ gf_int<BITS> clmul_most(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);




// special optimization where n = 2^k

constexpr bool is_power_of_2(int n) {
	return (n == 1) ? true : ( (n%2 == 0) ? is_power_of_2(n/2) : false );
}

template <int BITS, typename T, typename = bool_constant<true>>
struct clmul_underlying;

template <int BITS, typename T>
struct clmul_underlying< BITS, T, bool_constant<is_power_of_2(BITS)> > {
	__device__ inline static void reduce(T c[], int digit) {
		for (int stride = 2; stride <= BITS; stride *= 2) {
			if (digit % stride == 0) {
				c[digit] ^= c[digit + (stride >> 1)];
			}
			__syncthreads();
		}
	}
};

template <int BITS, typename T>
struct clmul_underlying< BITS, T, bool_constant<is_power_of_2(BITS) == false> > {
	__device__ inline static void reduce(T c[], int digit) {
		for (int stride = 2; stride <= BITS; stride *= 2) {
			int partner = digit + (stride >> 1);
			if (digit % stride == 0 && partner < BITS) {
				c[digit] ^= c[partner];
			}

			__syncthreads();
		}
	}
};



// tool functions for compile-time substitution

template <int BITS, typename T>
__device__ inline T a_mul_bi_least_half(T a, T b, int digit) {
	return a * (b & (1 << digit));
}

template <int BITS, typename T>
__device__ inline T a_mul_bi_most_half(T a, T b, int digit) {
	return (a >> (BITS - digit)) * ((b >> digit) & 1);
}



/*
template <int BITS, typename = bool_constant<true>>
struct constexpr_q_plus;

template <int BITS>
struct constexpr_q_plus??? {
	using double_t = typename gf_underlying<BITS * 2>::type;
	static const double_t value = (1 << BITS) | recur_q_plus<>::value;
}

template <using typename T,
	int BITS, T REMAINDER, int DIGIT,
	typename = bool_constant<true>>
struct recur_q_plus;

template <using typename T, int BITS, T REMAINDER, int DIGIT>
struct recur_q_plus<BITS, REMAINDER, DIGIT,
	bool_constant< (REMAINDER & (1<<(DIGIT-1))) != 0> > {
	value = (1<<(DIGIT-1)) | recur_q_plus<BITS, >
}
*/



// defination

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

	__device__ underlying_t getData() {
		constexpr underlying_t all_bits_true = static_cast<underlying_t>(-1);
		constexpr int underlying_bits = sizeof(underlying_t) * 8;
		return data & (all_bits_true >> (underlying_bits - BITS));
	}

	// addition

	__device__ gf_int & operator+=(const gf_int &rhs) {
		data ^= rhs.data;
		return *this;
	}

	friend __device__ gf_int operator+<BITS>(const gf_int &lhs, const gf_int &rhs);

	// carry-less multiplication
	// Warning: Better Reduction Algorithm?
	
	template <underlying_t (*f)(underlying_t a, underlying_t b, int digit)>
	__device__ gf_int & clmuled_by(const gf_int &rhs) {
		gf_int &lhs = *this;
	
		int digit = threadIdx.z;
		__shared__ underlying_t c[BITS];

		c[digit] = f(lhs.data, rhs.data, digit);
		__syncthreads();

		clmul_underlying<BITS, underlying_t>::reduce(c, digit);

		lhs.data = c[0];
		return lhs;
	}

	__device__ gf_int & clmuled_least_by(const gf_int &rhs) {
		return clmuled_by<a_mul_bi_least_half<BITS, underlying_t>>(rhs);
	}

	__device__ gf_int & clmuled_most_by(const gf_int &rhs) {
		return clmuled_by<a_mul_bi_most_half<BITS, underlying_t>>(rhs);
	}

	friend __device__ gf_int clmul_least<BITS>(const gf_int &lhs, const gf_int &rhs);
	friend __device__ gf_int clmul_most<BITS>(const gf_int &lhs, const gf_int &rhs);

	// Galois Field Multiplication
	// ... ...
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



// Type Aliases

using gf_int8_t = gf_int<8>;
using gf_int16_t = gf_int<16>;
using gf_int32_t = gf_int<32>;
using gf_int64_t = gf_int<64>;