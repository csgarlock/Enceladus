#include "../util.cuh"

#include <cuda_runtime.h>
#include <cmath>

struct Vector3 {
    
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    __unified_inlined__ Vector3 operator+(const Vector3 other) const { return Vector3{x + other.x, y + other.y, z + other.z}; };
    __unified_inlined__ Vector3& operator+=(const Vector3 other) { x = x+other.x; y = y+other.y; z = z+other.z; return *this; }

    __unified_inlined__ Vector3 operator-(const Vector3 other) const { return *this + -other; };
    __unified_inlined__ Vector3& operator-=(const Vector3 other) { *this += -other; return *this; }

    __unified_inlined__ Vector3 operator-() const { return Vector3{-x, -y, -z}; }

    __unified_inlined__ float& operator[](std::size_t idx) { return reinterpret_cast<float *>(this)[idx]; }
    __unified_inlined__ const float& operator[](std::size_t idx) const { return reinterpret_cast<const float *>(this)[idx]; }

    __unified_inlined__ float dot(const Vector3 other) const { return x*other.x + y*other.y + z*other.z; }

    __unified_inlined__ float length() const { return sqrtf(dot(*this)); } 
    __unified_inlined__ Vector3 normalize() const { float sf = 1.0f / length(); return Vector3{x * sf, y * sf, z * sf}; }

};