#include <cstdint>
#include <type_traits>

#include <iostream>
#include <iomanip>



constexpr int CURRENT_BITS = 8;

constexpr int BLOCK_DIM_X = 8;
constexpr int BLOCK_DIM_Y = (1024 / CURRENT_BITS / BLOCK_DIM_X);



template <bool IF>
using bool_constant = std::integral_constant<bool, IF>;



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



constexpr bool is_power_of_2(int n) {
	return (n == 1) ? true : ( (n%2 == 0) ? is_power_of_2(n/2) : false );
}

template <int BITS, typename = bool_constant<true>>
struct reduction;

template <int BITS>
struct reduction< BITS, bool_constant<is_power_of_2(BITS)> > {
	__host__ __device__ inline static bool index_check(const int digit, const int partner, const int stride) {
		return digit % stride == 0;
	}
};

template <int BITS>
struct reduction< BITS, bool_constant<is_power_of_2(BITS) == false> > {
	__host__ __device__ inline static bool index_check(const int digit, const int partner, const int stride) {
		return digit % stride == 0 && partner < BITS;
	}
};



template <int BITS, typename T>
__host__ __device__ inline constexpr T mask(const T memory) {
	constexpr int raw_len = sizeof(T) * 8;
    constexpr T mask_bits = static_cast<T>(-1) >> ((raw_len-BITS > 0) ? raw_len-BITS : 0);
	return memory & mask_bits;
}



template <int BITS>
__host__ __device__ inline constexpr
typename gf_raw<BITS>::type right_shift_in(const typename gf_raw<BITS>::type memory, const int offset) {
    return mask<BITS>(memory) >> offset;
}



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
struct constepxr_irreducible;



template <int BITS>
class gf_int;

template <int BITS>
__host__ __device__ gf_int<BITS> operator+(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);
template <int BITS>
__device__ gf_int<BITS> clmul_least(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);
template <int BITS>
__device__ gf_int<BITS> clmul_most(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs);

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
	static constexpr ext_t polynomial_divide(const ext_t poly, const int digit) {
		const ext_t bit_pick = right_shift_in<BITS*2>(poly, BITS) & (static_cast<ext_t>(1) << digit);
		const ext_t addition = irred_g * bit_pick;
		return ( bit_pick ? (static_cast<ext_t>(1)<<digit) : 0 ) |
			( (digit > 0) ? polynomial_divide(poly^addition, digit-1) : 0 );
	}
public:
	static constexpr ext_t irred_g = constepxr_irreducible<BITS>::polynomial;
	static constexpr raw_t g_star = static_cast<raw_t>( mask<BITS>(irred_g) );
	static constexpr ext_t q_plus = (static_cast<ext_t>(1) << BITS)
		| polynomial_divide(mask<BITS>(irred_g) << BITS, BITS - 1);
		// Use mask<>() to avoid integer overflow warning.

public:
	__host__ gf_int(): memory(0) {}
	__host__ __device__ gf_int(const raw_t raw_memory): memory(raw_memory) {}
	__host__ __device__ gf_int(const gf_int &old): memory(old.memory) {}

	__host__ __device__ constexpr raw_t value() const {
		return mask<BITS>(memory);
	}

	__device__ gf_int & operator=(const gf_int &rhs) {
		// __syncthreads();
		memory = rhs.memory;
		return *this;
	}
	__host__ gf_int & assigned(const gf_int &rhs) {
		memory = rhs.memory;
		return *this;
	}

	__device__ gf_int & operator+=(const gf_int &rhs) {
		// ****************************************
		// Begin Critical Section
		// Assumption 1: To threads in a bundle, all the accessible data are consistent

		memory ^= rhs.memory;
		// __syncthreads();
		// For any thread in bundle,
		// no read and no write before writing this down.
		
		// Assumption 1 is kept

		// End Critical Section
		// ****************************************

		return *this;
	}

	template <raw_t (*f_split)(const raw_t a, const raw_t b, const int digit)>
	__device__ gf_int & clmuled_by(const gf_int &rhs) {
		const gf_int &lhs = *this;
		// const int digit = threadIdx.z;

		// __shared__ raw_t all_c[BLOCK_DIM_X][BLOCK_DIM_Y][BITS];
		// auto &c = all_c[threadIdx.x][threadIdx.y];
		__shared__ raw_t all_c[BLOCK_DIM_X][BLOCK_DIM_Y];
		auto &c = all_c[threadIdx.x][threadIdx.y];

		c = raw_t(0);
		for (int i = 0; i < BITS; ++i) {
			c ^= f_split(lhs.memory, rhs.memory, i);
		}

		this->memory = c;
		// __syncthreads();

		return *this;
/*
		// ****************************************
		// Begin Critical Section

		c[digit] = f_split(lhs.memory, rhs.memory, digit);
		__syncthreads();

		// Warning: Better Reduction Algorithm?
		for (int stride = 2; stride <= BITS; stride *= 2) {
			int partner = digit + (stride >> 1);
			if (reduction<BITS>::index_check(digit, partner, stride)) {
				c[digit] ^= c[partner];
			}
			__syncthreads();
		}

		this->memory = c[0];
		__syncthreads();

		// End Critical Section
		// ****************************************
*/
	}

	__device__ gf_int & clmuled_least_by(const gf_int &rhs) {
		return clmuled_by<split_least_result<BITS>>(rhs);
	}

	__device__ gf_int & clmuled_most_by(const gf_int &rhs) {
		return clmuled_by<split_most_result<BITS>>(rhs);
	}

	// Galois Field Multiplication
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

	__device__ gf_int inverse() {
		gf_int c(*this);
		
		for (int i = 1; i < BITS-1; ++i) {
			// c := c^2 + x
			c *= c;
			c *= (*this);
		}
		c *= c;
		// x^(2^n-2) = x^(-1)
		// x^(2^n) = x^(1) as a cyclic group

		return c;
	}

};

template <int BITS>
__host__ __device__ gf_int<BITS> operator+(const gf_int<BITS> &lhs, const gf_int<BITS> &rhs)
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

template <int BITS>
__host__ std::ostream & operator<<(std::ostream &os, const gf_int<BITS> &rhs)
{
	return os << std::setfill('0')
		<< std::setw( (BITS/4 >= 1) ? (BITS/4) : 1 )
		<< +rhs.value();
}



template <>
struct constepxr_irreducible<4> {
	static constexpr typename gf_int<4>::ext_t polynomial = 0x13;
};
template <>
struct constepxr_irreducible<5> {
	static constexpr typename gf_int<5>::ext_t polynomial = 0x25;
};
template <>
struct constepxr_irreducible<8> {
	static constexpr typename gf_int<8>::ext_t polynomial = 0x11B;
};



// Type Aliases

using gf_int8_t = gf_int<8>;
using gf_int16_t = gf_int<16>;
using gf_int32_t = gf_int<32>;