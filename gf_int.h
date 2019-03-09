#include "xint.h"



// ****************************************
// class gf_int
// ****************************************

// Declaration of gf_int and its friend functions

template <int BITS>
class gf_int;

// special optimization where n = 2^k

constexpr bool is_power_of_2(int n) {
	return (n == 1) ? true : ( (n%2 == 0) ? is_power_of_2(n/2) : false );
}

template <int BITS, typename = bool_constant<true>>
struct reduce_underlying;

template <int BITS>
struct reduce_underlying< BITS, bool_constant<is_power_of_2(BITS)> > {
	__device__ inline static bool index_check(const int digit, const int partner, const int stride) {
		return digit % stride == 0;
	}
};

template <int BITS>
struct reduce_underlying< BITS, bool_constant<is_power_of_2(BITS) == false> > {
	__device__ inline static bool index_check(const int digit, const int partner, const int stride) {
		return digit % stride == 0 && partner < BITS;
	}
};

// Different functions for calculating
// Most and Least half of carry-less multiplication

template <int BITS, typename T>
__device__ inline T a_mul_bi_least_half(T a, T b, int digit) {
	return a * (b & (static_cast<T>(1) << digit));
}

template <int BITS, typename T>
__device__ inline T a_mul_bi_most_half(T a, T b, int digit) {
	return (a >> (BITS - digit)) * ((b >> digit) & static_cast<T>(1));
}



// Declaration: Irreducible Polynomials

template <int BITS>
struct constepxr_irreducible;

// Defination: Irreducible Polynomials

template <>
struct constepxr_irreducible<4> {
	static constexpr xint<4>::raw_t value() {
		return 0x13;
	}
};



// Defination

template <int BITS>
class gf_int
{
public:
	using std_t = xint<BITS>;
	using unsafe_t = typename std_t::raw_t;
	using ext_t = xint<BITS*2>;

private:
	std_t data;

private:
	static constexpr ext_t polynomial_divide(ext_t poly, int digit = BITS-1) {
		return ((((static_cast<ext_t>(1) << (BITS+digit)) & poly) != 0) ? (static_cast<ext_t>(1)<<digit) : 0) |
			polynomial_divide(poly ^ (irred_g * ((poly >> BITS) & (static_cast<ext_t>(1) << digit))), digit-1);

		/*constexpr bool flag = ((static_cast<ext_t>(1) << (BITS+digit)) & poly) != 0;
		poly = flag ? (poly ^ (irred_g << digit)) : poly;
		return (digit < BITS) ? 0 : (
			(flag ? (static_cast<ext_t>(1)<<digit) : 0) | polynomial_divide(poly, digit-1)
		);*/
	}
public:
	static constexpr ext_t irred_g = constepxr_irreducible<BITS>::value();
	static constexpr std_t g_star = static_cast<std_t>(irred_g);
	static constexpr ext_t q_plus = (static_cast<ext_t>(1) << BITS) |
		polynomial_divide(irred_g << BITS);

	__device__ unsafe_t test() const {
		return irred_g.value();
	}

public:
	__host__ __device__ constexpr gf_int(unsafe_t unsafe_data): data(unsafe_data) {}
	__host__ __device__ constexpr gf_int(const gf_int &old): data(old.data) {}

	__device__ constexpr unsafe_t value() const {
		return data.value();
	}

	__device__ constexpr gf_int & operator+=(const gf_int &rhs) {
		data ^= rhs.data;
		return *this;
	}

	// carry-less multiplication
	// Warning: Better Reduction Algorithm?
	
	template <std_t (*f)(std_t a, std_t b, int digit)>
	__device__ gf_int & clmuled_by(const gf_int &rhs) {
		gf_int &lhs = *this;
	
		int digit = threadIdx.z;
		__shared__ unsafe_t c[BITS];

		c[digit] = f(lhs.data, rhs.data, digit).value();
		__syncthreads();

		for (int stride = 2; stride <= BITS; stride *= 2) {
			int partner = digit + (stride >> 1);
			if (reduce_underlying<BITS>::index_check(digit, partner, stride)) {
				c[digit] ^= c[partner];
			}
			__syncthreads();
		}

		lhs.data = c[0];
		return lhs;
	}

	__device__ gf_int & clmuled_least_by(const gf_int &rhs) {
		return clmuled_by<a_mul_bi_least_half<BITS, std_t>>(rhs);
	}

	__device__ gf_int & clmuled_most_by(const gf_int &rhs) {
		return clmuled_by<a_mul_bi_most_half<BITS, std_t>>(rhs);
	}

	// Galois Field Multiplication
	// ... ...
};

template <int BITS>
__device__ constexpr gf_int<BITS> operator+(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs)
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