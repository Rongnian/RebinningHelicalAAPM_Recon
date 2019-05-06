#include "DD2MultiGPU.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <omp.h>

// Calculate the number of cores in the machine
#include <thread>
static const unsigned NUM_OF_CORES = std::thread::hardware_concurrency();

#ifndef __PI__
#define __PI__
#define TWOPI       6.283185307179586
#endif

#define FORCEINLINE 1
#if FORCEINLINE
#define INLINE __forceinline__
#else
#define INLINE inline
#endif

#if _DEBUG
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
// Same function as CUDA_CHECK_RETURN
#define CUDA_SAFE_CALL(call) do{ cudaError_t err = call; if (cudaSuccess != err) {  fprintf (stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err) );  exit(EXIT_FAILURE);  } } while (0)
#else
#define CUDA_CHECK_RETURN(value) {value;}
#define CUDA_SAFE_CALL(value) {value;}
#endif

#ifndef nullptr
#define nullptr NULL
#endif

INLINE __host__ __device__ const float2 operator/(const float2& a, float b) {
    return make_float2(a.x / b, a.y / b);
}
INLINE __host__ __device__ const float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
INLINE __host__ __device__ const float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
INLINE __host__ __device__ const double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
INLINE __host__ __device__ const float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
INLINE __host__ __device__ const float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
INLINE __host__ __device__ const float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
INLINE __host__ __device__ const float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
INLINE __host__ __device__ const float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
INLINE __host__ __device__ const double3 operator/(const double3& a, double b) {
    return make_double3(a.x / b, a.y / b, a.z / b);
}
INLINE __host__ __device__ const float3 operator-(const int3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
INLINE __host__ __device__ float length(const float2& a) {
    return sqrtf(a.x * a.x + a.y * a.y);
}
INLINE __host__ __device__ float length(const float3& a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
INLINE __host__ __device__ double length(const double3& a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
INLINE __host__ __device__ const float2 normalize(const float2& a) {
    return a / length(a);
}
INLINE __host__ __device__ const float3 normalize(const float3& a) {
    return a / length(a);
}
INLINE __host__ __device__ const double3 normalize(const double3& a) {
    return a / length(a);
}
INLINE __host__ __device__ float fminf(const float2& a) {
    return fminf(a.x, a.y);
}
INLINE __host__ __device__ float fminf(const float3& a) {
    return fminf(a.x, fminf(a.y, a.z));
}
INLINE __host__ __device__ float fmaxf(const float2& a) {
    return fmaxf(a.x, a.y);
}
INLINE __host__ __device__ float fmaxf(const float3& a) {
    return fmaxf(a.x, fmaxf(a.y, a.z));
}
INLINE __host__ __device__ const float3 fminf(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
INLINE __host__ __device__ const float3 fmaxf(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
INLINE __host__ __device__ const float2 fminf(const float2& a, const float2& b) {
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
INLINE __host__ __device__ const float2 fmaxf(const float2& a, const float2& b) {
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

// Linear interpolation
template<typename T>
INLINE __host__ __device__ T lerp(T v0, T v1, T t) {
    return fma(t, v1, fma(-t, v0, v0));
}

// Bi-linear interpolation
template<typename T>
INLINE __host__ __device__ T bilierp(T v0, T v1, T v2, T v3, T t1, T t2) {
    T vv0 = fma(t1, v1, fma(-t1, v0, v0));
    T vv1 = fma(t1, v3, fma(-t1, v2, v2));
    return fma(t2, vv1, fma(-t2, vv0, vv0));
}
INLINE __device__ double bilerp(int2 v0, int2 v1, int2 v2, int2 v3, double t1, double t2) {
    double v0_ = __hiloint2double(v0.y, v0.x);
    double v1_ = __hiloint2double(v1.y, v1.x);
    double v2_ = __hiloint2double(v2.y, v2.x);
    double v3_ = __hiloint2double(v3.y, v3.x);

    double vv0 = v0_ * (1.0 - t1) + v1_ * t1;
    double vv1 = v2_ * (1.0 - t1) + v3_ * t1;
    return vv0 * (1 - t2) + vv1 * t2;
}


// Whether a box intersect with a line
INLINE __host__ __device__ bool intersectBox(
    const float3& sour,
    const float3& dir,
    const float3& boxmin,
    const float3& boxmax,
    float* tnear, float* tfar)
{
    const float3 invR = make_float3(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    const float3 tbot = invR * (boxmin - sour);
    const float3 ttop = invR * (boxmax - sour);

    const float3 tmin = fminf(ttop, tbot);
    const float3 tmax = fmaxf(ttop, tbot);

    const float largest_tmin = fmaxf(tmin);
    const float smallest_tmax = fminf(tmax);
    *tnear = largest_tmin;
    *tfar = smallest_tmax;
    return smallest_tmax > largest_tmin;
}

template<typename T>
INLINE __host__ __device__ T regularizeAngle(T curang) {
    T c = curang;
    while (c >= TWOPI) { c -= TWOPI; }
    while (c < 0) { c += TWOPI; }
    return c;
}

INLINE __host__ __device__ void invRotVox(const float3& curVox,
    float3& virVox,
    const float2& cossinT,
    const float zP) {
    virVox.x = curVox.x * cossinT.x + curVox.y * cossinT.y;
    virVox.y = -curVox.x * cossinT.y + curVox.y * cossinT.x;
    virVox.z = curVox.z - zP;
}

INLINE __device__ float3 invRot(
    const float3 inV,
    const float2 cossin,
    const float zP) {
    float3 outV;
    outV.x = inV.x * cossin.x + inV.y * cossin.y;
    outV.x = -inV.x * cossin.y + inV.y * cossin.x;
    outV.z = inV.z - zP;
    return outV;
}


namespace CTMBIR
{
    struct Constant_MultiSlice
    {
        float x0;
        float y0;
        Constant_MultiSlice(const float _x0, const float _y0) :x0(_x0), y0(_y0) {}

        __device__ float2 operator()(const float& tp)
        {
            float curang = regularizeAngle(tp);
            return make_float2(cosf(curang), sinf(curang));

        }

    };

    template<typename T>
    struct ConstantForBackProjection {

        T x0;
        T y0;
        T z0;

        typedef thrust::tuple<T, T> InTuple;
        ConstantForBackProjection(const T _x0, const T _y0, const T _z0)
            : x0(_x0), y0(_y0), z0(_z0) {}

        __device__ float3 operator()(const InTuple& tp)
        {
            T curang = regularizeAngle(thrust::get<0>(tp));
            T zP = thrust::get<1>(tp);
            T cosT = cosf(curang);
            T sinT = sinf(curang);
            return make_float3(cosT, sinT, zP);
        }
    };


    template<>
    struct ConstantForBackProjection<double> {

        double x0;
        double y0;
        double z0;

        typedef thrust::tuple<double, double> InTuple;
        ConstantForBackProjection(const double _x0, const double _y0, const double _z0)
            : x0(_x0), y0(_y0), z0(_z0) {}

        __device__ double3 operator()(const InTuple& tp)
        {
            double curang = regularizeAngle(thrust::get<0>(tp));
            double zP = thrust::get<1>(tp);
            double cosT = cos(curang);
            double sinT = sin(curang);
            return make_double3(cosT, sinT, zP);
        }
    };

}


template<typename T>
void DD3Boundaries(int nrBoundaries, T*pCenters, T *pBoundaries)
{
    int i;
    if (nrBoundaries >= 3)
    {
        *pBoundaries++ = 1.5 * *pCenters - 0.5 * *(pCenters + 1);
        for (i = 1; i <= (nrBoundaries - 2); i++)
        {
            *pBoundaries++ = 0.5 * *pCenters + 0.5 * *(pCenters + 1);
            pCenters++;
        }
        *pBoundaries = 1.5 * *pCenters - 0.5 * *(pCenters - 1);
    }
    else
    {
        *pBoundaries = *pCenters - 0.5;
        *(pBoundaries + 1) = *pCenters + 0.5;
    }

}




///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
// Get one sub-volume from the whole volume.
// Assume that the volumes are stored in Z, X, Y order
template<typename T>
void getSubVolume(const T* vol,
    const size_t XN, const size_t YN, const size_t ZN,
    const size_t ZIdx_Start, const size_t ZIdx_End, T* subVol)
{
    const size_t SZN = ZIdx_End - ZIdx_Start;
    for (size_t yIdx = 0; yIdx != YN; ++yIdx)
    {
        for (size_t xIdx = 0; xIdx != XN; ++xIdx)
        {
            for (size_t zIdx = ZIdx_Start; zIdx != ZIdx_End; ++zIdx)
            {
                subVol[(yIdx * XN + xIdx) * SZN + (zIdx - ZIdx_Start)] = vol[(yIdx * XN + xIdx) * ZN + zIdx];
            }
        }
    }
}

template<typename T>
void getSubVolume(const T* vol,
    const size_t XYN, const size_t ZN,
    const size_t ZIdx_Start, const size_t ZIdx_End, T* subVol)
{
    const size_t SZN = ZIdx_End - ZIdx_Start;
    for (size_t xyIdx = 0; xyIdx != XYN; ++xyIdx)
    {
        for (size_t zIdx = ZIdx_Start; zIdx != ZIdx_End; ++zIdx)
        {
            subVol[xyIdx * SZN + (zIdx - ZIdx_Start)] = vol[xyIdx * ZN + zIdx];
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////

// For projection, before we divide the volume into serveral sub-volumes, we have
// to calculate the Z index range
template<typename T>
void getVolZIdxPair(const thrust::host_vector<T>& zPos, // Z position of the source.
                                                        //NOTE: We only assume the spiral CT case that zPos is increasing.
    const size_t PrjIdx_Start, const size_t PrjIdx_End,
    const T detCntIdxV, const T detStpZ, const int DNV,
    const T objCntIdxZ, const T dz, const int ZN, // Size of the volume
    int& ObjIdx_Start, int& ObjIdx_End) // The end is not included
{
    const T lowerPart = (detCntIdxV + 0.5) * detStpZ;
    const T upperPart = DNV * detStpZ - lowerPart;
    const T startPos = zPos[PrjIdx_Start] - lowerPart;
    const T endPos = zPos[PrjIdx_End - 1] + upperPart;

    ObjIdx_Start = floor((startPos / dz) + objCntIdxZ - 1);
    ObjIdx_End = ceil((endPos / dz) + objCntIdxZ + 1) + 1;

    ObjIdx_Start = (ObjIdx_Start < 0) ? 0 : ObjIdx_Start;
    ObjIdx_Start = (ObjIdx_Start > ZN) ? ZN : ObjIdx_Start;

    ObjIdx_End = (ObjIdx_End < 0) ? 0 : ObjIdx_End;
    ObjIdx_End = (ObjIdx_End > ZN) ? ZN : ObjIdx_End;
}

///////////////////////////////////////////////////////////////////////////////////
// For backprojection, after decide the subvolume range, we have to decide the
// projection range to cover the subvolume.
template<typename T>
void getPrjIdxPair(const thrust::host_vector<T>& zPos, // Z Position of the source.
                                                       // NOTE: we assume that it is pre sorted
    const size_t ObjZIdx_Start, const size_t ObjZIdx_End, // sub vol range,
                                                          // NOTE: the objZIdx_End is not included
    const T objCntIdxZ, const T dz, const int ZN,
    const T detCntIdxV, const T detStpZ, const int DNV,
    int& prjIdx_Start, int& prjIdx_End)
{
    const int PN = zPos.size();

    const T lowerPartV = (ObjZIdx_Start - objCntIdxZ - 0.5) * dz;
    const T highrPartV = lowerPartV + (ObjZIdx_End - ObjZIdx_Start) * dz;

    const T lowerPartDet = (detCntIdxV + 0.5) * detStpZ;
    const T upperPartDet = DNV * detStpZ - lowerPartDet;

    //The source position
    const T sourLPos = lowerPartV - upperPartDet;
    const T sourHPos = highrPartV + lowerPartDet;

    prjIdx_Start = thrust::upper_bound(zPos.begin(), zPos.end(), sourLPos) - zPos.begin() - 1;
    prjIdx_End = thrust::upper_bound(zPos.begin(), zPos.end(), sourHPos) - zPos.begin() + 2;
    prjIdx_Start = (prjIdx_Start < 0) ? 0 : prjIdx_Start;
    prjIdx_Start = (prjIdx_Start > PN) ? PN : prjIdx_Start;

    prjIdx_End = (prjIdx_End < 0) ? 0 : prjIdx_End;
    prjIdx_End = (prjIdx_End > PN) ? PN : prjIdx_End;
}


////////////////////////////////////////////////////////////////////////////////////
// The volume is also stored in Z, X, Y order
// Not tested yet.
template<typename T>
void combineVolume(
    T* vol, // The volume to be combined
    const int XN, const int YN, const int ZN,
    T** subVol, // All sub volumes
    const int* SZN, // Number of slices for each subVolume
    const int subVolNum) // Number of sub volumes
{
    int kk = 0;
    for (size_t yIdx = 0; yIdx != YN; ++yIdx)
    {
        for (size_t xIdx = 0; xIdx != XN; ++xIdx)
        {
            kk = 0;
            for (size_t volIdx = 0; volIdx != subVolNum; ++volIdx)
            {
                for (size_t zIdx = 0; zIdx != SZN[volIdx]; ++zIdx)
                {
                    vol[(yIdx * XN + xIdx) * ZN + kk] = subVol[volIdx][(yIdx * XN + xIdx) * SZN[volIdx] + zIdx];
                    kk = kk + 1;
                }
            }
        }
    }
}


#include <omp.h>
#define BLKX 32
#define BLKY 8
#define BLKZ 1

namespace DD2 {


    // Copy the volume from the original to
    template<typename Ta, typename Tb>
    __global__ void naive_copyToTwoVolumes(Ta* in_ZXY,
        Tb* out_ZXY, Tb* out_ZYX,
        int XN, int YN, int ZN)
    {
        int idz = threadIdx.x + blockIdx.x * blockDim.x;
        int idx = threadIdx.y + blockIdx.y * blockDim.y;
        int idy = threadIdx.z + blockIdx.z * blockDim.z;
        if (idx < XN && idy < YN && idz < ZN)
        {
            int i = (idy * XN + idx) * ZN + idz;
            int ni = (idy * (XN + 1) + (idx + 1)) * ZN + idz;
            int nj = (idx * (YN + 1) + (idy + 1)) * ZN + idz;

            out_ZXY[ni] = in_ZXY[i];
            out_ZYX[nj] = in_ZXY[i];
        }
    }


    __global__ void horizontalIntegral(float* prj, int DNU, int DNV, int PN)
    {
        int idv = threadIdx.x + blockIdx.x * blockDim.x;
        int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
        if (idv < DNV && pIdx < PN)
        {
            int headPrt = pIdx * DNU * DNV + idv;
            for (int ii = 1; ii < DNU; ++ii)
            {
                prj[headPrt + ii * DNV] = prj[headPrt + ii * DNV] + prj[headPrt + (ii - 1) * DNV];
            }
        }
    }

    void genSAT_for_Volume_MultiSlice(float* hvol,
        thrust::device_vector<float>&ZXY,
        thrust::device_vector<float>&ZYX,
        int XN, int YN, int ZN)
    {
        const int siz = XN * YN * ZN;

        thrust::device_vector<float> vol(hvol, hvol + siz);

        dim3 blk(64, 16, 1);
        dim3 gid(
            (ZN + blk.x - 1) / blk.x,
            (XN + blk.y - 1) / blk.y,
            (YN + blk.z - 1) / blk.z);

        naive_copyToTwoVolumes << <gid, blk >> >(
            thrust::raw_pointer_cast(&vol[0]),
            thrust::raw_pointer_cast(&ZXY[0]),
            thrust::raw_pointer_cast(&ZYX[0]),
            XN, YN, ZN);

        vol.clear();

        blk.x = 64;
        blk.y = 16;
        blk.z = 1;
        gid.x = (ZN + blk.x - 1) / blk.x;
        gid.y = (YN + blk.y - 1) / blk.y;
        gid.z = 1;

        horizontalIntegral << <gid, blk >> >(
            thrust::raw_pointer_cast(&ZXY[0]),
            XN + 1, ZN, YN);

        blk.x = 64;
        blk.y = 16;
        blk.z = 1;
        gid.x = (ZN + blk.x - 1) / blk.x;
        gid.y = (XN + blk.y - 1) / blk.y;
        gid.z = 1;

        horizontalIntegral << <gid, blk >> >(
            thrust::raw_pointer_cast(&ZYX[0]),
            YN + 1, ZN, XN);
    }


    __global__  void DD2_gpu_proj_branchless_sat2d_ker(
        cudaTextureObject_t volTex1,
        cudaTextureObject_t volTex2,
        float* proj,
        float2 s, // source position
        const float2* __restrict__ cossin,
        const float* __restrict__ xds,
        const float* __restrict__ yds,
        const float* __restrict__ bxds,
        const float* __restrict__ byds,
        float2 objCntIdx,
        float dx,
        int XN, int YN, int SLN,
        int DNU, int PN)
    {
        int slnIdx = threadIdx.x + blockIdx.x * blockDim.x;
        int detIdU = threadIdx.y + blockIdx.y * blockDim.y;
        int angIdx = threadIdx.z + blockIdx.z * blockDim.z;
        if (slnIdx < SLN && detIdU < DNU && angIdx < PN)
        {
            float2 dir = cossin[angIdx * SLN + slnIdx]; // cossin;

            float2 cursour = make_float2(
                s.x * dir.x - s.y * dir.y,
                s.x * dir.y + s.y * dir.x); // current source position;
            s = dir;

            float2 curDet = make_float2(
                xds[detIdU] * s.x - yds[detIdU] * s.y,
                xds[detIdU] * s.y + yds[detIdU] * s.x);

            float2 curDetL = make_float2(
                bxds[detIdU] * s.x - byds[detIdU] * s.y,
                bxds[detIdU] * s.y + byds[detIdU] * s.x);

            float2 curDetR = make_float2(
                bxds[detIdU + 1] * s.x - byds[detIdU + 1] * s.y,
                bxds[detIdU + 1] * s.y + byds[detIdU + 1] * s.x);

            dir = normalize(curDet - cursour);

            float factL = 0;
            float factR = 0;
            float constVal = 0;
            float obj = 0;
            float realL = 0;
            float realR = 0;
            float intersectLength = 0;

            float invdx = 1.0f / dx;
            //float summ[BLKX];
            float summ;
            if (fabsf(s.x) <= fabsf(s.y))
            {

                summ = 0;
                factL = (curDetL.y - cursour.y) / (curDetL.x - cursour.x);
                factR = (curDetR.y - cursour.y) / (curDetR.x - cursour.x);

                constVal = dx / fabsf(dir.x);
#pragma unroll
                for (int ii = 0; ii < XN; ++ii)
                {
                    obj = (ii - objCntIdx.x) * dx;

                    realL = (obj - curDetL.x) * factL + curDetL.y;
                    realR = (obj - curDetR.x) * factR + curDetR.y;

                    intersectLength = realR - realL;
                    realL = realL * invdx + objCntIdx.y + 1;
                    realR = realR * invdx + objCntIdx.y + 1;

                    summ += (tex3D<float>(volTex2, slnIdx + 0.5f, realR, ii + 0.5) - tex3D<float>(volTex2, slnIdx + 0.5, realL, ii + 0.5)) / intersectLength;

                }
                __syncthreads();
                proj[(angIdx * DNU + detIdU) * SLN + slnIdx] = summ * constVal;

            }
            else
            {

                summ = 0;
                factL = (curDetL.x - cursour.x) / (curDetL.y - cursour.y);
                factR = (curDetR.x - cursour.x) / (curDetR.y - cursour.y);

                constVal = dx / fabsf(dir.y);
#pragma unroll
                for (int ii = 0; ii < YN; ++ii)
                {
                    obj = (ii - objCntIdx.y) * dx;

                    realL = (obj - curDetL.y) * factL + curDetL.x;
                    realR = (obj - curDetR.y) * factR + curDetR.x;

                    intersectLength = realR - realL;
                    realL = realL * invdx + objCntIdx.x + 1;
                    realR = realR * invdx + objCntIdx.x + 1;

                    summ += (tex3D<float>(volTex1, slnIdx + 0.5f, realR, ii + 0.5) - tex3D<float>(volTex1, slnIdx + 0.5, realL, ii + 0.5)) / intersectLength;
                }
                __syncthreads();
                proj[(angIdx * DNU + detIdU) * SLN + slnIdx] = summ * constVal;
                //__syncthreads();
            }
        }
    }




    void DD2_gpu_proj_branchless_sat2d(
        float x0, float y0,
        int DNU,
        float* xds, float* yds,
        float imgXCenter, float imgYCenter,
        float* hangs, int PN,
        int XN, int YN, int SLN, // SLN is the slice number, it is the same as the rebinned projection slices
        float* vol, float* hprj, float dx, byte* mask, int gpunum)
    {
        for (int i = 0; i != XN * YN; ++i)
        {
            byte v = mask[i];
            for (int z = 0; z != SLN; ++z)
            {
                vol[i * SLN + z] = vol[i * SLN + z] * v;
            }
        }

        CUDA_SAFE_CALL(cudaSetDevice(gpunum));
        CUDA_SAFE_CALL(cudaDeviceReset());

        float* bxds = new float[DNU + 1];
        float* byds = new float[DNU + 1];

        DD3Boundaries(DNU + 1, xds, bxds);
        DD3Boundaries(DNU + 1, yds, byds);

        float objCntIdxX = (XN - 1.0) * 0.5 - imgXCenter / dx;
        float objCntIdxY = (YN - 1.0) * 0.5 - imgYCenter / dx;


        thrust::device_vector<float> SATZXY(SLN * (XN + 1) * YN, 0);
        thrust::device_vector<float> SATZYX(SLN * (YN + 1) * XN, 0);
        
        genSAT_for_Volume_MultiSlice(vol, SATZXY, SATZYX, XN, YN, SLN);

        cudaExtent volumeSize1;
        cudaExtent volumeSize2;
        assert(SLN > 0);
        volumeSize1.width = SLN;
        volumeSize1.height = XN + 1;
        volumeSize1.depth = YN;

        volumeSize2.width = SLN;
        volumeSize2.height = YN + 1;
        volumeSize2.depth = XN;

        cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc<float>();
        cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float>();

        cudaArray* d_volumeArray1 = nullptr;
        cudaArray* d_volumeArray2 = nullptr;

        CUDA_SAFE_CALL(cudaMalloc3DArray(&d_volumeArray1, &channelDesc1, volumeSize1));
        CUDA_SAFE_CALL(cudaMalloc3DArray(&d_volumeArray2, &channelDesc2, volumeSize2));

        cudaMemcpy3DParms copyParams1 = { 0 };
        copyParams1.srcPtr = make_cudaPitchedPtr((void*)
            thrust::raw_pointer_cast(&SATZXY[0]),
            volumeSize1.width * sizeof(float),
            volumeSize1.width, volumeSize1.height);
        copyParams1.dstArray = d_volumeArray1;
        copyParams1.extent = volumeSize1;
        copyParams1.kind = cudaMemcpyDeviceToDevice;

        cudaMemcpy3DParms copyParams2 = { 0 };
        copyParams2.srcPtr = make_cudaPitchedPtr((void*)
            thrust::raw_pointer_cast(&SATZYX[0]),
            volumeSize2.width * sizeof(float),
            volumeSize2.width, volumeSize2.height);
        copyParams2.dstArray = d_volumeArray2;
        copyParams2.extent = volumeSize2;
        copyParams2.kind = cudaMemcpyDeviceToDevice;

        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams1));
        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams2));

        SATZXY.clear();
        SATZYX.clear();

        cudaResourceDesc resDesc1;
        cudaResourceDesc resDesc2;
        memset(&resDesc1, 0, sizeof(resDesc1));
        memset(&resDesc2, 0, sizeof(resDesc2));

        resDesc1.resType = cudaResourceTypeArray;
        resDesc2.resType = cudaResourceTypeArray;

        resDesc1.res.array.array = d_volumeArray1;
        resDesc2.res.array.array = d_volumeArray2;

        cudaTextureDesc texDesc1;
        cudaTextureDesc texDesc2;

        memset(&texDesc1, 0, sizeof(texDesc1));
        memset(&texDesc2, 0, sizeof(texDesc2));

        texDesc1.addressMode[0] = cudaAddressModeClamp;
        texDesc1.addressMode[1] = cudaAddressModeClamp;
        texDesc1.addressMode[2] = cudaAddressModeClamp;

        texDesc2.addressMode[0] = cudaAddressModeClamp;
        texDesc2.addressMode[1] = cudaAddressModeClamp;
        texDesc2.addressMode[2] = cudaAddressModeClamp;

        texDesc1.filterMode = cudaFilterModeLinear;
        texDesc2.filterMode = cudaFilterModeLinear;

        texDesc1.readMode = cudaReadModeElementType;
        texDesc2.readMode = cudaReadModeElementType;

        texDesc1.normalizedCoords = false;
        texDesc2.normalizedCoords = false;

        cudaTextureObject_t texObj1 = 0;
        cudaTextureObject_t texObj2 = 0;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj1, &resDesc1, &texDesc1, nullptr));
        CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj2, &resDesc2, &texDesc2, nullptr));

        thrust::device_vector<float> prj(DNU * SLN * PN, 0);
        thrust::device_vector<float> d_xds(xds, xds + DNU);
        thrust::device_vector<float> d_yds(yds, yds + DNU);

        thrust::device_vector<float> d_bxds(bxds, bxds + DNU + 1);
        thrust::device_vector<float> d_byds(byds, byds + DNU + 1);
        thrust::device_vector<float> angs(hangs, hangs + PN * SLN);

        thrust::device_vector<float2> cossin(PN * SLN);
        thrust::transform(angs.begin(), angs.end(), cossin.begin(), CTMBIR::Constant_MultiSlice(x0, y0));


        dim3 blk(BLKX, BLKY, BLKZ);
        dim3 gid(
            (SLN + blk.x - 1) / blk.x,
            (DNU + blk.y - 1) / blk.y,
            (PN + blk.z - 1) / blk.z);

        DD2_gpu_proj_branchless_sat2d_ker << <gid, blk >> >(texObj1, texObj2,
            thrust::raw_pointer_cast(&prj[0]),
            make_float2(x0, y0),
            thrust::raw_pointer_cast(&cossin[0]),
            thrust::raw_pointer_cast(&d_xds[0]),
            thrust::raw_pointer_cast(&d_yds[0]),
            thrust::raw_pointer_cast(&d_bxds[0]),
            thrust::raw_pointer_cast(&d_byds[0]),
            make_float2(objCntIdxX, objCntIdxY), dx, XN, YN, SLN, DNU, PN);

        thrust::copy(prj.begin(), prj.end(), hprj);
        CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj1));
        CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj2));

        CUDA_SAFE_CALL(cudaFreeArray(d_volumeArray1));
        CUDA_SAFE_CALL(cudaFreeArray(d_volumeArray2));

        prj.clear();
        angs.clear();


        d_xds.clear();
        d_yds.clear();

        d_bxds.clear();
        d_byds.clear();
        cossin.clear();

        delete[] bxds;
        delete[] byds;

    }

}; //End NAMESPACE DD2


void DD2Proj_gpu(
    float x0, float y0,
    int DNU,
    float* xds, float* yds,
    float imgXCenter, float imgYCenter,
    float* hangs, int PN,
    int XN, int YN, int SLN,
    float* hvol, float* hprj,
    float dx,
    byte* mask, int gpunum)
{
    DD2::DD2_gpu_proj_branchless_sat2d(x0, y0, DNU, xds, yds, imgXCenter, imgYCenter,
        hangs, PN, XN, YN, SLN, hvol, hprj, dx, mask, gpunum);
}


#include <thread>

extern "C"
void DD2Proj_multiGPUs(float x0, float y0, int DNU,
    float* xds, float* yds, float imgXCenter, float imgYCenter,
    float* hangs, int PN, int XN, int YN, int SLN, float* hvol, float* hprj,
    float dx, byte* mask, int gpuNum, int* startIdx) 
{
    int* StartIdx = new int[gpuNum + 1];
    for (int i = 0; i < gpuNum; ++i) {
        StartIdx[i] = startIdx[i];
    }
    StartIdx[gpuNum] = SLN;
    // Divide the volume into several parts
    int* SLNn = new int[gpuNum];
    float** shvol = new float*[gpuNum];
    float** shprj = new float*[gpuNum];
    float** shang = new float*[gpuNum];
    for (int i = 0; i < gpuNum; ++i) {
        SLNn[i] = StartIdx[i + 1] - StartIdx[i];
        shvol[i] = new float[XN * YN * SLNn[i]];
        shprj[i] = new float[DNU * SLNn[i] * PN];
        shang[i] = new float[PN * SLNn[i]];
    }

    for (int pIdx = 0; pIdx != PN; ++pIdx)
    {
        for (int sIdx = 0; sIdx != SLN; ++sIdx)
        {
            for (int i = 0; i < gpuNum; ++i) {
                if (sIdx >= StartIdx[i] && sIdx < StartIdx[i + 1]) {
                    shang[i][pIdx * SLNn[i] + (sIdx - StartIdx[i])] = hangs[pIdx * SLN + sIdx];
                }
            }
        }
    }

    omp_set_num_threads(NUM_OF_CORES);
#pragma omp parallel for
    for (int i = 0; i < XN * YN; ++i)
    {
        for (int v = 0; v != SLN; ++v)
        {
            for (int mm = 0; mm < gpuNum; ++mm) {
                if (v >= StartIdx[mm] && v < StartIdx[mm + 1]) {
                    shvol[mm][i * SLNn[mm] + (v - startIdx[mm])] = hvol[i * SLN + v];
                }
            }
        }
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    omp_set_num_threads(gpuNum);

#pragma omp parallel for
    for (int i = 0; i < gpuNum; ++i)
    {
        DD2Proj_gpu(x0, y0, DNU, xds, yds,
            imgXCenter, imgYCenter, shang[i],
            PN, XN, YN, SLNn[i], shvol[i], shprj[i], dx, mask, i);
    }


    //Gather all
    omp_set_num_threads(NUM_OF_CORES);
#pragma omp parallel for
    for (int i = 0; i < DNU * PN; ++i)
    {
        for (int v = 0; v != SLN; ++v)
        {
            float val = 0;

            for (int mm = 0; mm < gpuNum; ++mm) {
                if (v >= StartIdx[mm] && v < StartIdx[mm+1]) {
                    val = shprj[mm][i * SLNn[mm] + (v - startIdx[mm])];
                }
            }
            hprj[i * SLN + v] = val;
        }
    }

    for (int mm = 0; mm < gpuNum; ++mm) {
        delete[] shprj[mm];
        delete[] shvol[mm];
    }
    delete[] shprj;
    delete[] shvol;
    delete[] SLNn;
    delete[] StartIdx;
}


#define BACK_BLKX 64
#define BACK_BLKY 4
#define BACK_BLKZ 1

enum BackProjectionMethod { _BRANCHLESS, _PSEUDODD, _ZLINEBRANCHLESS, _VOLUMERENDERING };

__global__ void addOneSidedZeroBoarder_multiSlice_Fan(const float* prj_in, float* prj_out, int DNU, int SLN, int PN)
{
    int idv = threadIdx.x + blockIdx.x * blockDim.x;
    int idu = threadIdx.y + blockIdx.y * blockDim.y;
    int pn = threadIdx.z + blockIdx.z * blockDim.z;
    if (idu < DNU && idv < SLN && pn < PN)
    {
        int i = (pn * DNU + idu) * SLN + idv;
        int ni = (pn * (DNU + 1) + (idu + 1)) * SLN + idv;
        prj_out[ni] = prj_in[i];
    }
}


__global__ void heorizontalIntegral_multiSlice_Fan(float* prj, int DNU, int SLN, int PN)
{
    int idv = threadIdx.x + blockIdx.x * blockDim.x;
    int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
    if (idv < SLN && pIdx < PN)
    {
        int headPrt = pIdx * DNU * SLN + idv;
        for (int ii = 1; ii < DNU; ++ii)
        {
            prj[headPrt + ii * SLN] = prj[headPrt + ii * SLN] + prj[headPrt + (ii - 1) * SLN];
        }
    }
}

thrust::device_vector<float> genSAT_of_Projection_multiSliceFan(
    float* hprj,
    int DNU, int SLN, int PN)
{
    const int siz = DNU * SLN * PN;
    const int nsiz = (DNU + 1) * SLN * PN;
    thrust::device_vector<float> prjSAT(nsiz, 0);
    thrust::device_vector<float> prj(hprj, hprj + siz);
    dim3 copyBlk(64, 16, 1); //MAY CHANGED
    dim3 copyGid(
        (SLN + copyBlk.x - 1) / copyBlk.x,
        (DNU + copyBlk.y - 1) / copyBlk.y,
        (PN + copyBlk.z - 1) / copyBlk.z);

    addOneSidedZeroBoarder_multiSlice_Fan << <copyGid, copyBlk >> >(
        thrust::raw_pointer_cast(&prj[0]),
        thrust::raw_pointer_cast(&prjSAT[0]),
        DNU, SLN, PN);

    const int nDNU = DNU + 1;

    copyBlk.x = 64; // MAY CHANGED
    copyBlk.y = 16;
    copyBlk.z = 1;
    copyGid.x = (SLN + copyBlk.x - 1) / copyBlk.x;
    copyGid.y = (PN + copyBlk.y - 1) / copyBlk.y;
    copyGid.z = 1;

    heorizontalIntegral_multiSlice_Fan << <copyGid, copyBlk >> >(
        thrust::raw_pointer_cast(&prjSAT[0]),
        nDNU, SLN, PN);

    return prjSAT;
}


void createTextureObject_multiSliceFan(
    cudaTextureObject_t& texObj,
    cudaArray* d_prjArray,
    int Width, int Height, int Depth,
    float* sourceData,
    cudaMemcpyKind memcpyKind,
    cudaTextureAddressMode addressMode,
    cudaTextureFilterMode textureFilterMode,
    cudaTextureReadMode textureReadMode,
    bool isNormalized)
{
    cudaExtent prjSize;
    prjSize.width = Width;
    prjSize.height = Height;
    prjSize.depth = Depth;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    cudaMalloc3DArray(&d_prjArray, &channelDesc, prjSize);
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(
        (void*)sourceData, prjSize.width * sizeof(float),
        prjSize.width, prjSize.height);
    copyParams.dstArray = d_prjArray;
    copyParams.extent = prjSize;
    copyParams.kind = memcpyKind;
    cudaMemcpy3D(&copyParams);
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_prjArray;
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    texDesc.addressMode[2] = addressMode;
    texDesc.filterMode = textureFilterMode;
    texDesc.readMode = textureReadMode;
    texDesc.normalizedCoords = isNormalized;

    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
}

void destroyTextureObject_multiSliceFan(cudaTextureObject_t& texObj, cudaArray* d_array)
{
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(d_array);
}




__global__ void DD2_gpu_back_ker_multiSlice_Fan(
    cudaTextureObject_t prjTexObj,
    float* vol,
    const byte* __restrict__ msk,
    const float2* __restrict__ cossin,
    float2 s,
    float S2D,
    float2 curvox, // imgCenter index
    float dx, float dbeta, float detCntIdx,
    int2 VN, int SLN, int PN)
{
    int3 id;
    id.z = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    id.x = threadIdx.y + __umul24(blockIdx.y, blockDim.y);
    id.y = threadIdx.z + __umul24(blockIdx.z, blockDim.z);
    if (id.z < SLN && id.x < VN.x && id.y < VN.y)
    {
        if (msk[id.y * VN.x + id.x] != 1)
        {
            return;
        }
        curvox = make_float2((id.x - curvox.x) * dx, (id.y - curvox.y) * dx);
        float2 cursour;
        float idxL, idxR;
        float cosVal;
        float summ = 0;

        float2 cossinT;
        float inv_sid = 1.0f / sqrtf(s.x * s.x + s.y * s.y);

        float2 dir;
        float l_square;
        float l;

        float alpha;
        float deltaAlpha;
        //S2D /= ddv;
        dbeta = 1.0 / dbeta;

        float ddv;
        for (int angIdx = 0; angIdx < PN; ++angIdx)
        {
            cossinT = cossin[angIdx * SLN + id.z];
            cursour = make_float2(
                s.x * cossinT.x - s.y * cossinT.y,
                s.x * cossinT.y + s.y * cossinT.x);

            dir = curvox - cursour;

            l_square = dir.x * dir.x + dir.y * dir.y;

            l = rsqrtf(l_square); // 1 / sqrt(l_square);
            alpha = asinf((cursour.y * dir.x - cursour.x * dir.y) * inv_sid * l);

            if (fabsf(cursour.x) > fabsf(cursour.y))
            {
                ddv = dir.x;
            }
            else
            {
                ddv = dir.y;
            }

            deltaAlpha = ddv / l_square * dx * 0.5;
            cosVal = dx / ddv * sqrtf(l_square);

            idxL = (alpha - deltaAlpha) * dbeta + detCntIdx + 1.0;
            idxR = (alpha + deltaAlpha) * dbeta + detCntIdx + 1.0;

            summ += (tex3D<float>(prjTexObj, id.z + 0.5, idxR, angIdx + 0.5) -
                tex3D<float>(prjTexObj, id.z + 0.5, idxL, angIdx + 0.5)) * cosVal;
        }
        __syncthreads();
        vol[(id.y * VN.x + id.x) * SLN + id.z] = summ;

    }
}


void DD2_gpu_back(float x0, float y0,
    int DNU,
    float* xds, float* yds,
    float detCntIdx,
    float imgXCenter, float imgYCenter,
    float* hangs, int PN, int XN, int YN, int SLN,
    float* hvol, float* hprj, float dx,
    byte* mask, int gpunum)
{
    cudaSetDevice(gpunum);
    cudaDeviceReset();

    float2 objCntIdx = make_float2(
        (XN - 1.0) * 0.5 - imgXCenter / dx,
        (YN - 1.0) * 0.5 - imgYCenter / dx); // set the center of the image
    float2 sour = make_float2(x0, y0);

    thrust::device_vector<byte> msk(mask, mask + XN * YN);
    thrust::device_vector<float> vol(XN * YN * SLN, 0);

    const float S2D = hypotf(xds[0] - x0, yds[0] - y0);

    thrust::device_vector<float2> cossin(PN * SLN);
    thrust::device_vector<float> angs(hangs, hangs + PN * SLN);
    thrust::transform(angs.begin(), angs.end(), cossin.begin(), CTMBIR::Constant_MultiSlice(x0, y0));

    //Calculate the corresponding parameters such as
    // return make_float4(detCtrIdxU, detCtrIdxV, dbeta, ddv);
    float detTransverseSize = sqrt(powf(xds[1] - xds[0], 2) + powf(yds[1] - yds[0], 2));
    float dbeta = atanf(detTransverseSize / S2D * 0.5) * 2.0f;

    cudaArray* d_prjArray = nullptr;
    cudaTextureObject_t texObj;

    dim3 blk;
    dim3 gid;

    thrust::device_vector<float> prjSAT;

    //Generate the SAT along XY direction;
    prjSAT = genSAT_of_Projection_multiSliceFan(hprj, DNU, SLN, PN);
    createTextureObject_multiSliceFan(texObj, d_prjArray, SLN, DNU + 1, PN,
        thrust::raw_pointer_cast(&prjSAT[0]),
        cudaMemcpyDeviceToDevice,
        cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeElementType, false);
    prjSAT.clear();

    blk.x = BACK_BLKX; // May be changed
    blk.y = BACK_BLKY;
    blk.z = BACK_BLKZ;
    gid.x = (SLN + blk.x - 1) / blk.x;
    gid.y = (XN + blk.y - 1) / blk.y;
    gid.z = (YN + blk.z - 1) / blk.z;

    DD2_gpu_back_ker_multiSlice_Fan << <gid, blk >> >(texObj,
        thrust::raw_pointer_cast(&vol[0]),
        thrust::raw_pointer_cast(&msk[0]),
        thrust::raw_pointer_cast(&cossin[0]),
        make_float2(x0, y0),
        S2D,
        make_float2(objCntIdx.x, objCntIdx.y),
        dx, dbeta, detCntIdx, make_int2(XN, YN), SLN, PN);

    thrust::copy(vol.begin(), vol.end(), hvol);
    destroyTextureObject_multiSliceFan(texObj, d_prjArray);

    vol.clear();
    msk.clear();
    angs.clear();
    cossin.clear();


}

void DD2Back_gpu(
    float x0, float y0,
    int DNU,
    float* xds, float* yds,
    float detCntIdx,
    float imgXCenter, float imgYCenter,
    float* hangs, int PN,
    int XN, int YN, int SLN,
    float* hvol, float* hprj,
    float dx,
    byte* mask, int gpunum)
{

    DD2_gpu_back(x0, y0, DNU, xds, yds, detCntIdx, imgXCenter, imgYCenter,
        hangs, PN, XN, YN, SLN, hvol, hprj, dx, mask, gpunum);

}

extern "C"
void DD2Back_multiGPUs(
    float x0, float y0,
    int DNU,
    float* xds, float* yds,
    float detCntIdx,
    float imgXCenter, float imgYCenter,
    float* hangs, int PN,
    int XN, int YN, int SLN,
    float* hvol, float* hprj,
    float dx,
    byte* mask, int gpuNum, int* startIdx) {

    int* StartIdx = new int[gpuNum + 1];
    for (int i = 0; i < gpuNum; ++i) {
        StartIdx[i] = startIdx[i];
    }
    StartIdx[gpuNum] = SLN;
    // Divide the volume into several parts
    int* SLNn = new int[gpuNum];
    float** shvol = new float*[gpuNum];
    float** shprj = new float*[gpuNum];
    float** shang = new float*[gpuNum];
    for (int i = 0; i < gpuNum; ++i) {
        SLNn[i] = StartIdx[i + 1] - StartIdx[i];
        shvol[i] = new float[XN * YN * SLNn[i]];
        shprj[i] = new float[DNU * SLNn[i] * PN];
        shang[i] = new float[PN * SLNn[i]];
    }

    for (int pIdx = 0; pIdx != PN; ++pIdx)
    {
        for (int sIdx = 0; sIdx != SLN; ++sIdx)
        {
            for (int i = 0; i < gpuNum; ++i) {
                if (sIdx >= StartIdx[i] && sIdx < StartIdx[i + 1]) {
                    shang[i][pIdx * SLNn[i] + (sIdx - StartIdx[i])] = hangs[pIdx * SLN + sIdx];
                }
            }
        }
    }

    omp_set_num_threads(NUM_OF_CORES);
#pragma omp parallel for
    for (int i = 0; i < DNU * PN; ++i)
    {
        for (int v = 0; v != SLN; ++v)
        {
            for (int mm = 0; mm < gpuNum; ++mm) {
                if (v >= StartIdx[mm] && v < StartIdx[mm + 1]) {
                    shprj[mm][i * SLNn[mm] + (v - StartIdx[mm])] = hprj[i * SLN + v];
                }
            }
        }
    }

    cudaDeviceSynchronize();
    omp_set_num_threads(gpuNum);
#pragma omp parallel for
    for (int i = 0; i < gpuNum; ++i)
    {
        DD2Back_gpu(x0, y0, DNU, xds, yds, detCntIdx,
            imgXCenter, imgYCenter, shang[i],
            PN, XN, YN, SLNn[i], shvol[i], shprj[i], dx, mask, i);
    }


    //Gather all
    omp_set_num_threads(NUM_OF_CORES);
#pragma omp parallel for
    for (int i = 0; i < XN * YN; ++i)
    {
        for (int v = 0; v != SLN; ++v)
        {
            float val = 0;
            for (int mm = 0; mm < gpuNum; ++mm) {
                if (v >= StartIdx[mm] && v < StartIdx[mm+1])
                {
                    val = shvol[mm][i * SLNn[mm] + (v - startIdx[mm])];
                }
            }
            hvol[i * SLN + v] = val;
        }
    }

    for (int i = 0; i < gpuNum; ++i) {
        delete[] shprj[i];
        delete[] shvol[i];
    }
    
    delete[] shprj;
    delete[] shvol;
    delete[] SLNn;
    delete[] StartIdx;
}

namespace AAPM {
    template<typename T>
    const T mod(const T& lambda, const T& regV)
    {
        T v = lambda;
        while (v > regV)
        {
            v -= regV;
        }
        return v;
    }
}

#ifndef M_PI
#define M_PI (3.14159265358979323846264)
#endif

extern "C"
void HelicalToFan(
    float* Proj,  					// rebinned projection; in order (Channel, View Index, Slice Index)  TODO: need permute after call by MATLAB
    float* Views, 					// rebinned views (View Index, Slice Index) TODO: need permute after call by MATLAB
    float* proj,  					// raw projection data In order : (Height Index, Channel Index, View Index(Total View))
    float* zPos,  					// sampling position
    const int SLN,                  // slice number
    const float SD, 				// source to detector distance
    const float SO,					// source to iso-center distance
    const float BVAngle,        	// The begin view
    const int DetWidth,         	// number of detector columns
    const int DetHeight,        	// number of detector rows
    const float PerDetW,        	// Detector cell size along channel direction
    const float PerDetH,        	// Detector cell size along bench moving direction
    const int DefTimes,         	// Number of views per rotation
    const float DetCenterW,     	// Detector Center Index
    const float SpiralPitchFactor 	// Pitch defined in SIEMENS
)
{
    // Calculate necessary parameters
    const float PerDetA = 2.0f * atanf(PerDetW / SD * 0.5f);
    const float delta = 2.0 * M_PI / DefTimes;
    const float DetCenterHreal = (DetHeight - 1.0) * 0.5 / PerDetH;
    const float h = SpiralPitchFactor * PerDetH * DetHeight * SO / SD;
    const float deltaZ = h / DefTimes;

    omp_set_num_threads(NUM_OF_CORES);
#pragma omp parallel for
    for (int iii = 0; iii < SLN; iii++)
    {
        float z = zPos[iii];
        float* PData = new float[DefTimes * DetWidth];
        int* mark = new int[DefTimes * DetWidth];
        float* ViewAngles = new float[DefTimes];

        std::fill(PData, PData + DefTimes * DetWidth, 0.0f);
        std::fill(mark, mark + DefTimes * DetWidth, 0.0f);
        int start = floorf((z - h * 0.5f) / deltaZ);
        std::fill(ViewAngles, ViewAngles + DefTimes, 0.0f);
        for (size_t i = 1; i <= start + DefTimes - 1; ++i)
        {
            float* data = proj + (i - 1) * DetWidth * DetHeight; // In order (Height Idx, Channel Idx)

            if (i >= start)
            {
                float zz = (i - 1) * deltaZ;
                float lambda = (i - 1) * delta + BVAngle;
                float lambdaFan = AAPM::mod<float>(lambda, 2 * M_PI);
                int ibeta = i - start + 1;
                ViewAngles[ibeta - 1] = lambdaFan;

                for (size_t j = 1; j <= DetWidth; ++j)
                {
                    float ang = (j - DetCenterW) * PerDetA; //
                    float SM = SO * cosf(ang); //[Noo, 1999, Single-Slice rebinning method]
                    float tanAngV = (z - zz) / SM;
                    float v = SD * tanAngV; // curve detector
                    float dv = (DetCenterHreal + v) / PerDetH + 1.0;
                    int idv = floorf(dv);
                    float t = dv - idv;
                    float cosAngV = SD / sqrtf(SD * SD + v * v);
                    // --------------------- linear interpolation
                    float temp = 0;
                    int tempmark = 0;

                    if ((idv >= 1) && (idv <= DetHeight) &&
                        (idv + 1 >= 1) && (idv + 1 <= DetHeight))
                    {
                        temp = data[(j - 1) * DetHeight + (idv - 1)] * (1 - t)
                            + data[(j - 1) * DetHeight + idv] * t; // problem here
                        tempmark = 1;
                    }
                    if (((ibeta - 1) * DetWidth + (j - 1)) > DefTimes * DetWidth)
                    {
                        std::cout << ibeta << " " << j << "\n";
                    }
                    // In order (Channel Index, View Index);
                    PData[(ibeta - 1) * DetWidth + (j - 1)] = cosAngV * temp;
                    mark[(ibeta - 1) * DetWidth + (j - 1)] = tempmark;
                } // end for j
            }// end if
        }// end for i

        int numOfab = 0;
        for (size_t i = 1; i <= DefTimes; ++i)
        {
            for (size_t j = 1; j <= DetWidth; ++j)
            {
                if (mark[(i - 1) * DetWidth + (j - 1)] == 0)
                {
                    numOfab = numOfab + 1;
                    std::fill(mark + (i - 1) * DetWidth, mark + i * DetWidth, 0);
                    std::fill(PData + (i - 1) * DetWidth, PData + i * DetWidth, 0);
                    break;
                }
            }
        }

        float ViewAngle = 360 * (DefTimes - numOfab) / DefTimes;
        std::cout << "The total rotation angle (degree) = " << iii << std::endl;

        for (size_t angIdx = 0; angIdx != DefTimes; ++angIdx)
        {
            for (size_t detIdx = 0; detIdx != DetWidth; ++detIdx)
            {
                Proj[(iii * DefTimes + angIdx) * DetWidth + detIdx] =
                    PData[angIdx * DetWidth + detIdx];
            }
            Views[iii * DefTimes + angIdx] = ViewAngles[angIdx];
        }
        delete[] PData;
        delete[] mark;
        delete[] ViewAngles;
    }
}
