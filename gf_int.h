#include <cstdint>
#include <type_traits>

#include <iostream>
#include <iomanip>



/* ************************************************** */
/* Compile-time Types and Expressions */



// Bool Trait

template <bool IF>
using bool_constant = std::integral_constant<bool, IF>;



// Underlying Type of Galois-field-integer

constexpr bool in_closed_range(const int val, const int l, const int r)
{
	return l <= val && val <= r;
}

template <int BITS, typename = bool_constant<true>>
struct gf_raw;

template <int BITS>
struct gf_raw<BITS, bool_constant<in_closed_range(BITS, 1, 8)>> {
	using type = uint8_t;
};

template <int BITS>
struct gf_raw<BITS, bool_constant<in_closed_range(BITS, 9, 16)>> {
	using type = uint16_t;
};

template <int BITS>
struct gf_raw<BITS, bool_constant<in_closed_range(BITS, 17, 32)>> {
	using type = uint32_t;
};

template <int BITS>
struct gf_raw<BITS, bool_constant<in_closed_range(BITS, 33, 64)>> {
	using type = uint64_t;
};



// Primary Polynomial over GF(2^n)
// The defination is after gf_int's.

template <int BITS>
struct gf_constants;



/* ************************************************** */
/* Constexpr Functions */



// Mask the leftmost bits, and leave the significant ones 
// for gf_int<BITS>'s underlying type.

template <typename T>
__host__ __device__ inline constexpr int _bitwise() {
	return sizeof(T) * 8;
}

template <int BITS, typename T>
__host__ __device__ inline constexpr T _mask_code() {
	return static_cast<T>(-1) >> ((_bitwise<T>()-BITS > 0) ? _bitwise<T>()-BITS : 0);
}

template <int BITS, typename T>
__host__ __device__ inline constexpr T mask(const T memory) {
	return memory & _mask_code<BITS, T>();
}



// A Safe Right-shift
// Since the leftmost bits of gf_raw<BITS>::type are undefined,
// right-shift needs an extra mask operation.

template <int BITS>
__host__ __device__ inline constexpr
typename gf_raw<BITS>::type right_shift_in(const typename gf_raw<BITS>::type memory, const int offset) {
    return mask<BITS>(memory) >> offset;
}



/* ************************************************** */
/* gf_int and its tool functions */



// How to split gf_int<BITS> in its carry-less multiplication.

template <int BITS>
__host__ __device__ inline typename gf_raw<BITS>::type split_least_result
	(const typename gf_raw<BITS>::type a, const typename gf_raw<BITS>::type b, const int digit) {
	using T = typename gf_raw<BITS>::type;
	return a * (b & (static_cast<T>(1) << digit));
}

template <int BITS>
__host__ __device__ inline typename gf_raw<BITS>::type split_most_result
	(const typename gf_raw<BITS>::type a, const typename gf_raw<BITS>::type b, const int digit) {
	using T = typename gf_raw<BITS>::type;
	return right_shift_in<BITS>(a, BITS-digit) * (right_shift_in<BITS>(b, digit) & static_cast<T>(1));
}



template <int BITS>
class gf_int;

template <int BITS>
__host__ __device__ gf_int<BITS> operator+(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);
template <int BITS>
__device__ gf_int<BITS> clmul_least(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);
template <int BITS>
__device__ gf_int<BITS> clmul_most(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);
template <int BITS>
__device__ gf_int<BITS> operator*(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);



template <int BITS>
class gf_int
{
public:
	using raw_t = typename gf_raw<BITS>::type;
	using ext_t = typename gf_raw<BITS * 2>::type;

private:
	raw_t memory;
	// Reading [memory] directly is unsafe,
	// because we never know what the leftmost bits are,
	// after the unpredictable preceding operations.
	// 
	// When simply writing to [memory], it doesn't matter whether you mask the [memory] or not.
	// But when reading, do it by value().

private:
	static constexpr ext_t _bit_pick(const ext_t poly, const int digit) {
		return right_shift_in<BITS*2>(poly, BITS) & (static_cast<ext_t>(1) << digit);
	}
	static constexpr ext_t _addition(const ext_t poly, const int digit) {
		return prim_g * _bit_pick(poly, digit);
	}
	static constexpr ext_t _polynomial_divide(const ext_t poly, const int digit) {
		return ( _bit_pick(poly, digit) ? (static_cast<ext_t>(1)<<digit) : 0 ) |
			( (digit > 0) ? _polynomial_divide(poly^_addition(poly,digit), digit-1) : 0 );
	}

public:
	static constexpr ext_t prim_g = gf_constants<BITS>::prim_poly;
	static constexpr raw_t g_star = static_cast<raw_t>( mask<BITS>(prim_g) );
	static constexpr ext_t q_plus = (static_cast<ext_t>(1) << BITS)
		| _polynomial_divide(mask<BITS>(prim_g) << BITS, BITS - 1);
		// Use mask<>() to avoid integer overflow warning.

public:
	__host__ gf_int(): memory(0) {}
	__host__ __device__ gf_int(const raw_t raw_memory): memory(raw_memory) {}
	__host__ __device__ gf_int(const gf_int &old): memory(old.memory) {}

	__host__ __device__ raw_t value() const {
		return mask<BITS>(memory);
	}

	__host__ __device__ gf_int & operator=(const gf_int &rhs) {
		memory = rhs.memory;
		return *this;
	}

	__device__ gf_int & operator+=(const gf_int &rhs) {
		memory ^= rhs.memory;
		return *this;
	}

	template <raw_t (*f_split)(const raw_t a, const raw_t b, const int digit)>
	__device__ gf_int & clmuled_by(const gf_int &rhs) {
		const gf_int &lhs = *this;

		raw_t c(0);
		for (int i = 0; i < BITS; ++i) {
			c ^= f_split(lhs.memory, rhs.memory, i);
		}

		this->memory = c;

		return *this;
	}

	__device__ gf_int & clmuled_least_by(const gf_int &rhs) {
		return clmuled_by<split_least_result<BITS>>(rhs);
	}

	__device__ gf_int & clmuled_most_by(const gf_int &rhs) {
		return clmuled_by<split_most_result<BITS>>(rhs);
	}

	__device__ gf_int & operator*=(const gf_int &rhs) {
		gf_int c_least = clmul_least(*this, rhs);
		gf_int &c = *this;
		c.clmuled_most_by(rhs);

		c += clmul_most<BITS>(c, mask<BITS>(q_plus));
		// The most significant bit of q_plus is always 1.
		// So the result should be:
		// (q_plus[0...BITS-1] * c)'s most significant part, PLUS a copy of c.

		c.clmuled_least_by(g_star);

		c += c_least;
		return *this;
	}

	__device__ gf_int inverse() const {
		gf_int c(*this);
		
		for (int i = 1; i < BITS-1; ++i) {
			// c := c^2 + x
			c *= c;
			c *= (*this);
		}
		c *= c;
		// For a cyclic group
		// x^(-1) = x^(2^n-2)

		return c;
	}

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

template <int BITS>
__device__ gf_int<BITS> operator*(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs)
{
	gf_int<BITS> result(lhs);
	result *= rhs;
	return result;
}



// Output in Host

template <int BITS>
__host__ std::ostream & operator<<(std::ostream &os, const gf_int<BITS> &rhs)
{
	return os << std::setfill('0')
		<< std::setw( (BITS/4 >= 1) ? (BITS/4) : 1 )
		<< +rhs.value();
}



// Primary Polynomial over GF(2^n)

template <>
struct gf_constants<4> {
	static constexpr typename gf_int<4>::ext_t prim_poly = 0x13;
};
template <>
struct gf_constants<8> {
	static constexpr typename gf_int<8>::ext_t prim_poly = 0x11B;
};
template <>
struct gf_constants<12> {
	static constexpr typename gf_int<12>::ext_t prim_poly = 010123;
};
template <>
struct gf_constants<16> {
	static constexpr typename gf_int<16>::ext_t prim_poly = 0210013;
};
template <>
struct gf_constants<20> {
	static constexpr typename gf_int<20>::ext_t prim_poly = 04000011;
};
template <>
struct gf_constants<24> {
	static constexpr typename gf_int<24>::ext_t prim_poly = 0100000207;
};
template <>
struct gf_constants<28> {
	static constexpr typename gf_int<28>::ext_t prim_poly = 02000000011;
};
template <>
struct gf_constants<32> {
	static constexpr typename gf_int<32>::ext_t prim_poly = 040020000007;
};



// Type Aliases

using gf_int8_t = gf_int<8>;
using gf_int16_t = gf_int<16>;
using gf_int32_t = gf_int<32>;