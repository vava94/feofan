//
// Created by vok on 19.08.2020.
//
#pragma once

#include <cuda_runtime.h>


#include <string>

#ifdef __linux__
#include <glob.h>
#endif

typedef unsigned char uchar;

/// cudaVectorTypeInfo
template<class T> struct cudaVectorTypeInfo;

template<> struct cudaVectorTypeInfo<uchar>  { typedef uint8_t Base; };
template<> struct cudaVectorTypeInfo<uchar3> { typedef uint8_t Base; };
template<> struct cudaVectorTypeInfo<uchar4> { typedef uint8_t Base; };

template<> struct cudaVectorTypeInfo<float>  { typedef float Base; };
template<> struct cudaVectorTypeInfo<float3> { typedef float Base; };
template<> struct cudaVectorTypeInfo<float4> { typedef float Base; };

/// make_vec
template<typename T> struct cuda_assert_false : std::false_type { };
template<typename T> inline __host__ __device__ T make_vec( typename cudaVectorTypeInfo<T>::Base x, typename cudaVectorTypeInfo<T>::Base y, typename cudaVectorTypeInfo<T>::Base z, typename cudaVectorTypeInfo<T>::Base w )	{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }

template<> inline __host__ __device__ uchar  make_vec( uint8_t x, uint8_t y, uint8_t z, uint8_t w )	{ return x; }
template<> inline __host__ __device__ uchar3 make_vec( uint8_t x, uint8_t y, uint8_t z, uint8_t w )	{ return make_uchar3(x,y,z); }
template<> inline __host__ __device__ uchar4 make_vec( uint8_t x, uint8_t y, uint8_t z, uint8_t w )	{ return make_uchar4(x,y,z,w); }

template<> inline __host__ __device__ float  make_vec( float x, float y, float z, float w )		{ return x; }
template<> inline __host__ __device__ float3 make_vec( float x, float y, float z, float w )		{ return make_float3(x,y,z); }
template<> inline __host__ __device__ float4 make_vec( float x, float y, float z, float w )		{ return make_float4(x,y,z,w); }

/// iDivUp
inline __device__ __host__ unsigned int iDivUp(unsigned int a, unsigned int b ) { return (a % b != 0) ? (a / b + 1) : (a / b); }

/// alpha
template<typename T> inline __device__ typename cudaVectorTypeInfo<T>::Base alpha( T vec, typename cudaVectorTypeInfo<T>::Base default_alpha=255 )	{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }

template<> inline __host__ __device__ uint8_t alpha( uchar3 vec, uint8_t default_alpha )		{ return default_alpha; }
template<> inline __host__ __device__ uint8_t alpha( uchar4 vec, uint8_t default_alpha )		{ return vec.w; }

template<> inline __host__ __device__ float alpha( float3 vec, float default_alpha )			{ return default_alpha; }
template<> inline __host__ __device__ float alpha( float4 vec, float default_alpha )			{ return vec.w; }

// cast_vec<T> templates
template<typename T> inline __host__ __device__ T cast_vec( const uchar3& a )				{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }
template<typename T> inline __host__ __device__ T cast_vec( const uchar4& a )				{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }
template<typename T> inline __host__ __device__ T cast_vec( const float3& a )				{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }
template<typename T> inline __host__ __device__ T cast_vec( const float4& a )				{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }

template<> inline __host__ __device__ uchar3 cast_vec( const uchar3& a )					{ return make_uchar3(a.x, a.y, a.z); }
template<> inline __host__ __device__ uchar4 cast_vec( const uchar3& a )					{ return make_uchar4(a.x, a.y, a.z, 0); }
template<> inline __host__ __device__ float3 cast_vec( const uchar3& a )					{ return make_float3(float(a.x), float(a.y), float(a.z)); }
template<> inline __host__ __device__ float4 cast_vec( const uchar3& a )					{ return make_float4(float(a.x), float(a.y), float(a.z), 0.0f); }

template<> inline __host__ __device__ uchar3 cast_vec( const uchar4& a )					{ return make_uchar3(a.x, a.y, a.z); }
template<> inline __host__ __device__ uchar4 cast_vec( const uchar4& a )					{ return make_uchar4(a.x, a.y, a.z, a.w); }
template<> inline __host__ __device__ float3 cast_vec( const uchar4& a )					{ return make_float3(float(a.x), float(a.y), float(a.z)); }
template<> inline __host__ __device__ float4 cast_vec( const uchar4& a )					{ return make_float4(float(a.x), float(a.y), float(a.z), float(a.w)); }

template<> inline __host__ __device__ uchar3 cast_vec( const float3& a )					{ return make_uchar3(a.x, a.y, a.z); }
template<> inline __host__ __device__ uchar4 cast_vec( const float3& a )					{ return make_uchar4(a.x, a.y, a.z, 0); }
template<> inline __host__ __device__ float3 cast_vec( const float3& a )					{ return make_float3(a.x, a.y, a.z); }
template<> inline __host__ __device__ float4 cast_vec( const float3& a )					{ return make_float4(a.x, a.y, a.z, 0.0f); }

template<> inline __host__ __device__ uchar3 cast_vec( const float4& a )					{ return make_uchar3(a.x, a.y, a.z); }
template<> inline __host__ __device__ uchar4 cast_vec( const float4& a )					{ return make_uchar4(a.x, a.y, a.z, a.w); }
template<> inline __host__ __device__ float3 cast_vec( const float4& a )					{ return make_float3(a.x, a.y, a.z); }
template<> inline __host__ __device__ float4 cast_vec( const float4& a )					{ return make_float4(a.x, a.y, a.z, a.w); }

bool cudaAllocMapped(void **cpuPtr, void **gpuPtr, size_t size);