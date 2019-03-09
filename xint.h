#include <cstdint>
#include <type_traits>



// ****************************************
// Type Traits in Compile-time
// ****************************************

// Bool Trait
template <bool IF>
using bool_constant = std::integral_constant<bool, IF>;

// constexpr mathematical and logic functions
constexpr bool in_closed_range(const int val, const int l, const int r)
{
	return l <= val && val <= r;
}



template <int BITS, typename = bool_constant<true>>
struct x_underlying;

template <int BITS>
struct x_underlying<BITS, bool_constant<in_closed_range(BITS, 1, 8)>> {
	using type = uint8_t;
};

template <int BITS>
struct x_underlying<BITS, bool_constant<in_closed_range(BITS, 9, 16)>> {
	using type = uint16_t;
};

template <int BITS>
struct x_underlying<BITS, bool_constant<in_closed_range(BITS, 17, 32)>> {
	using type = uint32_t;
};

template <int BITS>
struct x_underlying<BITS, bool_constant<in_closed_range(BITS, 33, 64)>> {
	using type = uint64_t;
};



template <int BITS> class xint;

template <int BITS>
class xint {
public:
    using raw_t = typename x_underlying<BITS>::type;

private:
    // Vaild Access: data[0...(BITS-1)]
    // Invaild Access: data[...0] data[BITS...]
    raw_t data;

    // VA: value
    // IA: Undefined
    __host__ __device__ raw_t raw() const {
        return data;
    }

public:
    // VA: value
    // IA: data[BITS...END] = 0 else Undefined
    __host__ __device__ raw_t value() const {
        constexpr int raw_bits = sizeof(raw_t) * 8;
        constexpr raw_t mask = static_cast<raw_t>(-1) >> (raw_bits - BITS);
        return data & mask;
    }

public:
    __host__ __device__ xint(): xint(0) {}

    // VA: raw value's [0...(BITS-1)]
    // IA: Undefined
    __host__ __device__ xint(const raw_t &raw_data): data(raw_data) {}

    // VA: old xint's value
    // IA: Undefined
    __host__ __device__ xint(const xint &old): data(old.data) {}

    // VA: when become longer, is padding 0s + old value
    //     OR cutted old value + keeped part
    // IA: Undefined
    template <int TO_BITS>
    __host__ __device__ xint<TO_BITS> toBits() const {
        using new_raw_t = typename xint<TO_BITS>::raw_t;
        return xint<TO_BITS>( static_cast<new_raw_t>(value()) );
    }

    __host__ __device__ xint & operator+=(const xint &rhs) {
        data += rhs.raw();
        return *this;
    }

    __host__ __device__ xint & operator*=(const xint &rhs) {
        data *= rhs.raw();
        return *this;
    }

    __host__ __device__ xint & operator<<=(const int &rhs) {
        data <<= rhs;
        return *this;
    }

    __host__ __device__ xint & operator>>=(const int &rhs) {
        data = this->value() >> rhs;
        return *this;
    }

    __host__ __device__ xint & operator&=(const xint &rhs) {
        data &= rhs.raw();
        return *this;
    }

    __host__ __device__ xint & operator|=(const xint &rhs) {
        data |= rhs.raw();
        return *this;
    }

    __host__ __device__ xint & operator^=(const xint &rhs) {
        data ^= rhs.raw();
        return *this;
    }

    __host__ __device__ explicit operator bool() const {
        return value();
    }
};

template <int BITS>
__host__ __device__ xint<BITS> operator+(const xint<BITS> &lhs, const xint<BITS> &rhs)
{
    xint<BITS> result(lhs);
    result += rhs;
    return result;
}

template <int BITS>
__host__ __device__ xint<BITS> operator*(const xint<BITS> &lhs, const xint<BITS> &rhs)
{
    xint<BITS> result(lhs);
    result *= rhs;
    return result;
}

template <int BITS>
__host__ __device__ xint<BITS> operator<<(const xint<BITS> &lhs, const int &rhs)
{
    xint<BITS> result(lhs);
    result <<= rhs;
    return result;
}

template <int BITS>
__host__ __device__ xint<BITS> operator>>(const xint<BITS> &lhs, const int &rhs)
{
    xint<BITS> result(lhs);
    result >>= rhs;
    return result;
}

template <int BITS>
__host__ __device__ xint<BITS> operator&(const xint<BITS> &lhs, const xint<BITS> &rhs)
{
    xint<BITS> result(lhs);
    result &= rhs;
    return result;
}

template <int BITS>
__host__ __device__ xint<BITS> operator|(const xint<BITS> &lhs, const xint<BITS> &rhs)
{
    xint<BITS> result(lhs);
    result |= rhs;
    return result;
}

template <int BITS>
__host__ __device__ xint<BITS> operator^(const xint<BITS> &lhs, const xint<BITS> &rhs)
{
    xint<BITS> result(lhs);
    result ^= rhs;
    return result;
}
