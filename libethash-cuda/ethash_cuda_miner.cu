/*
  This file is part of c-ethash.

  c-ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  c-ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** 
* @author Tim Hughes <tim@twistedfury.com>
* @date 2015
*
* CUDA port by Grigore Lupescu <grigore.lupescu@gmail.com>
*/

#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <queue>
#include <vector>
#include <libethash/util.h>
#include <libethash/ethash.h>
#include <string>
#include "ethash_cuda_miner.h"


#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define ETHASH_BYTES 32

#undef min
#undef max

#define DAG_SIZE 262688
#define MAX_OUTPUTS 63
#define GROUP_SIZE 64
#define ACCESSES 64

/* APPLICATION SETTINGS */
#define __USE_SHARED_MEMORY__	1
#define __ENDIAN_LITTLE__	1

#define THREADS_PER_HASH (8)
#define HASHES_PER_LOOP (GROUP_SIZE / THREADS_PER_HASH)

#define FNV_PRIME	0x01000193

__device__ __constant__ uint2 Keccak_f1600_RC[24] = {
	{0x00000001, 0x00000000},
	{0x00008082, 0x00000000},
	{0x0000808a, 0x80000000},
	{0x80008000, 0x80000000},
	{0x0000808b, 0x00000000},
	{0x80000001, 0x00000000},
	{0x80008081, 0x80000000},
	{0x00008009, 0x80000000},
	{0x0000008a, 0x00000000},
	{0x00000088, 0x00000000},
	{0x80008009, 0x00000000},
	{0x8000000a, 0x00000000},
	{0x8000808b, 0x00000000},
	{0x0000008b, 0x80000000},
	{0x00008089, 0x80000000},
	{0x00008003, 0x80000000},
	{0x00008002, 0x80000000},
	{0x00000080, 0x80000000},
	{0x0000800a, 0x00000000},
	{0x8000000a, 0x80000000},
	{0x80008081, 0x80000000},
	{0x00008080, 0x80000000},
	{0x80000001, 0x00000000},
	{0x80008008, 0x80000000},
};

inline __host__ __device__ uint bitselect(uint a, uint b, uint c)
{
	return (~c & a) | (c & b);
}

inline __host__ __device__ uint2 bitselect(uint2 a, uint2 b, uint2 c)
{
	uint2 res;
	res.x = (~(c.x) & a.x) | (c.x & b.x);
	res.y = (~(c.y) & a.y) | (c.y & b.y);

	return res;
}

inline __host__ __device__ uint2 operator^(uint2 a, uint2 b)
{
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

/*----------------------------------------------------------------------
*	HOST/TARGET FUNCTIONS
*---------------------------------------------------------------------*/

/******************************************
* FUNCTION: keccak_f1600_round
*******************************************/
__device__ void keccak_f1600_round(__restrict__ uint2* a, uint r, uint out_size)
{
	#if !__ENDIAN_LITTLE__
		for (uint i = 0; i != 25; ++i)
			a[i] = make_uint2(a[i].y, a[i].x);
	#endif

	uint2 b[25];
	uint2 t;

	// Theta
	b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
	b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
	b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
	b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
	b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
	t = b[4] ^ make_uint2(b[1].x << 1 | b[1].y >> 31, b[1].y << 1 | b[1].x >> 31);
	a[0] = a[0] ^ t;
	a[5] = a[5] ^ t;
	a[10] = a[10] ^ t;
	a[15] = a[15] ^ t;
	a[20] = a[20] ^ t;
	t = b[0] ^ make_uint2(b[2].x << 1 | b[2].y >> 31, b[2].y << 1 | b[2].x >> 31);
	a[1] = a[1] ^ t;
	a[6] = a[6] ^ t;
	a[11] = a[11] ^ t;
	a[16] = a[16] ^ t;
	a[21] = a[21] ^ t;
	t = b[1] ^ make_uint2(b[3].x << 1 | b[3].y >> 31, b[3].y << 1 | b[3].x >> 31);
	a[2] = a[2] ^ t;
	a[7] = a[7] ^ t;
	a[12] = a[12] ^ t;
	a[17] = a[17] ^ t;
	a[22] = a[22] ^ t;
	t = b[2] ^ make_uint2(b[4].x << 1 | b[4].y >> 31, b[4].y << 1 | b[4].x >> 31);
	a[3] = a[3] ^ t;
	a[8] = a[8] ^ t;
	a[13] = a[13] ^ t;
	a[18] = a[18] ^ t;
	a[23] = a[23] ^ t;
	t = b[3] ^ make_uint2(b[0].x << 1 | b[0].y >> 31, b[0].y << 1 | b[0].x >> 31);
	a[4] = a[4] ^ t;
	a[9] = a[9] ^ t;
	a[14] = a[14] ^ t;
	a[19] = a[19] ^ t;
	a[24] = a[24] ^ t;

	// Rho Pi
	b[0] = a[0];
	b[10] = make_uint2(a[1].x << 1 | a[1].y >> 31, a[1].y << 1 | a[1].x >> 31);
	b[7] = make_uint2(a[10].x << 3 | a[10].y >> 29, a[10].y << 3 | a[10].x >> 29);
	b[11] = make_uint2(a[7].x << 6 | a[7].y >> 26, a[7].y << 6 | a[7].x >> 26);
	b[17] = make_uint2(a[11].x << 10 | a[11].y >> 22, a[11].y << 10 | a[11].x >> 22);
	b[18] = make_uint2(a[17].x << 15 | a[17].y >> 17, a[17].y << 15 | a[17].x >> 17);
	b[3] = make_uint2(a[18].x << 21 | a[18].y >> 11, a[18].y << 21 | a[18].x >> 11);
	b[5] = make_uint2(a[3].x << 28 | a[3].y >> 4, a[3].y << 28 | a[3].x >> 4);
	b[16] = make_uint2(a[5].y << 4 | a[5].x >> 28, a[5].x << 4 | a[5].y >> 28);
	b[8] = make_uint2(a[16].y << 13 | a[16].x >> 19, a[16].x << 13 | a[16].y >> 19);
	b[21] = make_uint2(a[8].y << 23 | a[8].x >> 9, a[8].x << 23 | a[8].y >> 9);
	b[24] = make_uint2(a[21].x << 2 | a[21].y >> 30, a[21].y << 2 | a[21].x >> 30);
	b[4] = make_uint2(a[24].x << 14 | a[24].y >> 18, a[24].y << 14 | a[24].x >> 18);
	b[15] = make_uint2(a[4].x << 27 | a[4].y >> 5, a[4].y << 27 | a[4].x >> 5);
	b[23] = make_uint2(a[15].y << 9 | a[15].x >> 23, a[15].x << 9 | a[15].y >> 23);
	b[19] = make_uint2(a[23].y << 24 | a[23].x >> 8, a[23].x << 24 | a[23].y >> 8);
	b[13] = make_uint2(a[19].x << 8 | a[19].y >> 24, a[19].y << 8 | a[19].x >> 24);
	b[12] = make_uint2(a[13].x << 25 | a[13].y >> 7, a[13].y << 25 | a[13].x >> 7);
	b[2] = make_uint2(a[12].y << 11 | a[12].x >> 21, a[12].x << 11 | a[12].y >> 21);
	b[20] = make_uint2(a[2].y << 30 | a[2].x >> 2, a[2].x << 30 | a[2].y >> 2);
	b[14] = make_uint2(a[20].x << 18 | a[20].y >> 14, a[20].y << 18 | a[20].x >> 14);
	b[22] = make_uint2(a[14].y << 7 | a[14].x >> 25, a[14].x << 7 | a[14].y >> 25);
	b[9] = make_uint2(a[22].y << 29 | a[22].x >> 3, a[22].x << 29 | a[22].y >> 3);
	b[6] = make_uint2(a[9].x << 20 | a[9].y >> 12, a[9].y << 20 | a[9].x >> 12);
	b[1] = make_uint2(a[6].y << 12 | a[6].x >> 20, a[6].x << 12 | a[6].y >> 20);

	// Chi
	a[0] = bitselect(b[0] ^ b[2], b[0], b[1]);
	a[1] = bitselect(b[1] ^ b[3], b[1], b[2]);
	a[2] = bitselect(b[2] ^ b[4], b[2], b[3]);
	a[3] = bitselect(b[3] ^ b[0], b[3], b[4]);
	if (out_size >= 4)
	{
		a[4] = bitselect(b[4] ^ b[1], b[4], b[0]);
		a[5] = bitselect(b[5] ^ b[7], b[5], b[6]);
		a[6] = bitselect(b[6] ^ b[8], b[6], b[7]);
		a[7] = bitselect(b[7] ^ b[9], b[7], b[8]);
		a[8] = bitselect(b[8] ^ b[5], b[8], b[9]);
		if (out_size >= 8)
		{
			a[9] = bitselect(b[9] ^ b[6], b[9], b[5]);
			a[10] = bitselect(b[10] ^ b[12], b[10], b[11]);
			a[11] = bitselect(b[11] ^ b[13], b[11], b[12]);
			a[12] = bitselect(b[12] ^ b[14], b[12], b[13]);
			a[13] = bitselect(b[13] ^ b[10], b[13], b[14]);
			a[14] = bitselect(b[14] ^ b[11], b[14], b[10]);
			a[15] = bitselect(b[15] ^ b[17], b[15], b[16]);
			a[16] = bitselect(b[16] ^ b[18], b[16], b[17]);
			a[17] = bitselect(b[17] ^ b[19], b[17], b[18]);
			a[18] = bitselect(b[18] ^ b[15], b[18], b[19]);
			a[19] = bitselect(b[19] ^ b[16], b[19], b[15]);
			a[20] = bitselect(b[20] ^ b[22], b[20], b[21]);
			a[21] = bitselect(b[21] ^ b[23], b[21], b[22]);
			a[22] = bitselect(b[22] ^ b[24], b[22], b[23]);
			a[23] = bitselect(b[23] ^ b[20], b[23], b[24]);
			a[24] = bitselect(b[24] ^ b[21], b[24], b[20]);
		}
	}
	

	// Iota
	a[0] = a[0] ^ Keccak_f1600_RC[r];

	#if !__ENDIAN_LITTLE__
		for (uint i = 0; i != 25; ++i)
			a[i] = make_uint2(a[i].y, a[i].x);
	#endif
}

/******************************************
* FUNCTION: keccak_f1600_no_absorb
*******************************************/
__device__ void keccak_f1600_no_absorb(ulong* a, uint in_size, uint out_size, uint isolate)
{
	
	for (uint i = in_size; i != 25; ++i)
	{
		a[i] = 0;
	}

#if __ENDIAN_LITTLE__
	a[in_size] = a[in_size] ^ 0x0000000000000001;
	a[24-out_size*2] = a[24-out_size*2] ^ 0x8000000000000000;
#else
	a[in_size] = a[in_size] ^ 0x0100000000000000;
	a[24-out_size*2] = a[24-out_size*2] ^ 0x0000000000000080;
#endif

	// Originally I unrolled the first and last rounds to interface
	// better with surrounding code, however I haven't done this
	// without causing the AMD compiler to blow up the VGPR usage.
	uint r = 0;
	do
	{
		// This dynamic branch stops the AMD compiler unrolling the loop
		// and additionally saves about 33% of the VGPRs, enough to gain another
		// wavefront. Ideally we'd get 4 in flight, but 3 is the best I can
		// massage out of the compiler. It doesn't really seem to matter how
		// much we try and help the compiler save VGPRs because it seems to throw
		// that information away, hence the implementation of keccak here
		// doesn't bother.
		if (isolate)
		{
			keccak_f1600_round((uint2*)a, r++, 25);
		}
	}
	while (r < 23);

	// final round optimised for digest size
	keccak_f1600_round((uint2*)a, r++, out_size);
}

#define copy(dst, src, count) for (uint i = 0; i != count; ++i) { (dst)[i] = (src)[i]; }

#define countof(x) (sizeof(x) / sizeof(x[0]))

__device__ uint fnv(uint x, uint y)
{
	return x * FNV_PRIME ^ y;
}

__device__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 res;

	res.x = a.x * FNV_PRIME ^ b.x;
	res.y = a.y * FNV_PRIME ^ b.y;
	res.z = a.z * FNV_PRIME ^ b.z;
	res.w = a.w * FNV_PRIME ^ b.w;

	return res;
}

__device__ uint fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

typedef union
{
	ulong ulongs[32 / sizeof(ulong)];
	uint uints[32 / sizeof(uint)];
} hash32_t;

typedef union
{
	ulong ulongs[64 / sizeof(ulong)];
	uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union
{
	uint uints[128 / sizeof(uint)];
	uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

/******************************************
* FUNCTION: init_hash
*******************************************/
__device__ hash64_t init_hash( hash32_t const* header, ulong nonce, uint isolate)
{
	hash64_t init;
	uint const init_size = countof(init.ulongs);
	uint const hash_size = countof(header->ulongs);

	// sha3_512(header .. nonce)
	ulong state[25];
	copy(state, header->ulongs, hash_size);
	state[hash_size] = nonce;
	keccak_f1600_no_absorb(state, hash_size + 1, init_size, isolate);

	copy(init.ulongs, state, init_size);
	return init;
}

/******************************************
* FUNCTION: inner_loop
*******************************************/
__device__ uint inner_loop(uint4 init, uint thread_id, uint* share, hash128_t const* g_dag, uint isolate)
{
	uint4 mix = init;

	// share init0
	if (thread_id == 0)
		*share = mix.x;

	__syncthreads();
	__threadfence_block();
	uint init0 = *share;

	uint a = 0;
	do
	{
		bool update_share = thread_id == (a/4) % THREADS_PER_HASH;

		#pragma unroll
		for (uint i = 0; i != 4; ++i)
		{
			if (update_share)
			{
				uint m[4] = { mix.x, mix.y, mix.z, mix.w };
				*share = fnv(init0 ^ (a+i), m[i]) % DAG_SIZE;
			}
			__syncthreads();
			__threadfence_block();

			mix = fnv4(mix, g_dag[*share].uint4s[thread_id]);
		}
	}
	while ((a += 4) != (ACCESSES & isolate));

	return fnv_reduce(mix);
}

/******************************************
* FUNCTION: final_hash
*******************************************/
__device__ hash32_t final_hash(hash64_t const* init, hash32_t const* mix, uint isolate)
{
	ulong state[25];

	hash32_t hash;
	uint const hash_size = countof(hash.ulongs);
	uint const init_size = countof(init->ulongs);
	uint const mix_size = countof(mix->ulongs);

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->ulongs, init_size);
	copy(state + init_size, mix->ulongs, mix_size);
	keccak_f1600_no_absorb(state, init_size+mix_size, hash_size, isolate);

	// copy out
	copy(hash.ulongs, state, hash_size);
	return hash;
}


typedef union
{
	struct
	{
		hash64_t init;
		uint pad; // avoid lds bank conflicts
	};
	hash32_t mix;
} compute_hash_share;

/******************************************
* FUNCTION: compute_hash_simple
* INFO: shared memory optimisations
*******************************************/
__device__ hash32_t compute_hash(
	compute_hash_share* share,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong nonce,
	uint isolate
	)
{
	uint const gid = blockIdx.x * blockDim.x + threadIdx.x;
	hash32_t mix;
	hash64_t init;

	// Compute one init hash per work item.
	init = init_hash(g_header, nonce, isolate);
	
/* check if Keccak works */
#if 0
	mix.uints[0] = 0;
	return final_hash(&init, &mix, isolate);
#endif	

	// Threads work together in this phase in groups of 8.
	uint const thread_id = gid % THREADS_PER_HASH;
	uint const hash_id = (gid % GROUP_SIZE) / THREADS_PER_HASH;

	uint i = 0;
	do
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;
		__threadfence_block();

		uint4 thread_init = share[hash_id].init.uint4s[thread_id % (64 / sizeof(uint4))];
		__threadfence_block();

		uint thread_mix = inner_loop(thread_init, thread_id, share[hash_id].mix.uints, g_dag, isolate);

		share[hash_id].mix.uints[thread_id] = thread_mix;
		__threadfence_block();

		if (i == thread_id)
			mix = share[hash_id].mix;
		__threadfence_block();
	}
	while (++i != (THREADS_PER_HASH & isolate));

	return final_hash(&init, &mix, isolate);
}

/******************************************
* FUNCTION: compute_hash_simple
* INFO: no optimisations
*******************************************/
__device__ hash32_t compute_hash_simple(
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong nonce,
	uint isolate
	)
{
	hash64_t init = init_hash(g_header, nonce, isolate);

	hash128_t mix;
	for (uint i = 0; i != countof(mix.uint4s); ++i)
	{
		mix.uint4s[i] = init.uint4s[i % countof(init.uint4s)];
	}
	
	uint mix_val = mix.uints[0];
	uint init0 = mix.uints[0];
	uint a = 0;
	do
	{
		uint pi = fnv(init0 ^ a, mix_val) % DAG_SIZE;
		uint n = (a+1) % countof(mix.uints);

		for (uint i = 0; i != countof(mix.uints); ++i)
		{
			mix.uints[i] = fnv(mix.uints[i], g_dag[pi].uints[i]);
			mix_val = i == n ? mix.uints[i] : mix_val;
		}
	}
	while (++a != (ACCESSES & isolate));

	// reduce to output
	hash32_t fnv_mix;
	for (uint i = 0; i != countof(fnv_mix.uints); ++i)
	{
		fnv_mix.uints[i] = fnv_reduce(mix.uint4s[i]);
	}
	
	return final_hash(&init, &fnv_mix, isolate);
}

/******************************************
* FUNCTION: ethash_hash
*******************************************/
__global__ void ethash_hash(
	hash32_t* g_hashes,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	uint isolate
	)
{

#if __USE_SHARED_MEMORY__
	__shared__ compute_hash_share share[HASHES_PER_LOOP];

	uint const gid = blockIdx.x * blockDim.x + threadIdx.x;
	g_hashes[gid] = compute_hash(share, g_header, g_dag, start_nonce + gid, isolate);
#else
	uint const gid = blockIdx.x * blockDim.x + threadIdx.x;
	g_hashes[gid] = compute_hash_simple(g_header, g_dag, start_nonce + gid, isolate);
#endif
}

/******************************************
* FUNCTION: ethash_search
*******************************************/
__global__ void ethash_search(
	uint* g_output,
	 hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
	)
{
#if __USE_SHARED_MEMORY__
	__shared__ compute_hash_share share[HASHES_PER_LOOP];

	uint const gid = (blockIdx.x + gridDim.x  * blockIdx.y) * blockDim.x + threadIdx.x + blockDim.x * threadIdx.y;
	hash32_t hash = compute_hash(share, g_header, g_dag, start_nonce + gid, isolate);
#else
	uint const gid = (blockIdx.x + gridDim.x  * blockIdx.y) * blockDim.x + threadIdx.x + blockDim.x * threadIdx.y;
	hash32_t hash = compute_hash(share, g_header, g_dag, start_nonce + gid, isolate);
#endif

	if (hash.ulongs[countof(hash.ulongs)-1] < target)
	{
		uint slot = min(MAX_OUTPUTS, atomicInc(&g_output[0], 1) + 1);
		g_output[slot] = gid;
	}
}

using namespace std;

//ethash_cuda_miner::search_hook::~search_hook() {}


/*----------------------------------------------------------------------
*	HOST ONLY FUNCTIONS
*---------------------------------------------------------------------*/


ethash_cuda_miner::ethash_cuda_miner()
{
}

ethash_cuda_miner::~ethash_cuda_miner()
{
	finish();
}

std::string ethash_cuda_miner::platform_info(unsigned _platformId, unsigned _deviceId)
{
	/*
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty())
	{
		cout << "No OpenCL platforms found." << endl;
		return std::string();
	}

	// get GPU device of the selected platform
	std::vector<cl::Device> devices;
	unsigned platform_num = std::min<unsigned>(_platformId, platforms.size() - 1);
	platforms[platform_num].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.empty())
	{
		cout << "No OpenCL devices found." << endl;
		return std::string();
	}

	// use selected default device
	unsigned device_num = std::min<unsigned>(_deviceId, devices.size() - 1);
	cl::Device& device = devices[device_num];
	std::string device_version = device.getInfo<CL_DEVICE_VERSION>();

	return "{ \"platform\": \"" + platforms[platform_num].getInfo<CL_PLATFORM_NAME>() + "\", \"device\": \"" + device.getInfo<CL_DEVICE_NAME>() + "\", \"version\": \"" + device_version + "\" }";
	*/

    cudaDeviceProp prop;
	cudaGetDeviceProperties (&prop, _deviceId );
	return "{ \"platform\": \"NVidia Cuda\", \"device\": \"" + string(prop.name) + "\", \"version\":\"unknown\" }";
}

unsigned ethash_cuda_miner::get_num_platforms()
{
	/*std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	return platforms.size();
	*/
	return 1;
}

unsigned ethash_cuda_miner::get_num_devices(unsigned _platformId)
{
	/*
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty())
	{
		cout << "No OpenCL platforms found." << endl;
		return 0;
	}

	std::vector<cl::Device> devices;
	unsigned platform_num = std::min<unsigned>(_platformId, platforms.size() - 1);
	platforms[platform_num].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.empty())
	{
		cout << "No OpenCL devices found." << endl;
		return 0;
	}
	return devices.size();
	*/
	int count;
	cudaGetDeviceCount (&count );
	return count; 
}


void ethash_cuda_miner::finish()
{
}

/******************************************
* FUNCTION: init
*******************************************/
__host__ bool ethash_cuda_miner::init(uint8_t const* _dag, uint64_t _dagSize, unsigned workgroup_size, unsigned _platformId, unsigned _deviceId) {
//bool ethash_cuda_miner::init(ethash_params const& params, ethash_h256_t const *seed, unsigned workgroup_size)
	// store params
	//m_params = params;
	cudaError r;

	unsigned num_devices = get_num_devices();
	if (_deviceId >= num_devices) {
		cout << "deviceId " << _deviceId << " does not exist, defaulting to highest deviceId available" << num_devices - 1 << "." << endl;
	}
    cout << "deviceId:" << _deviceId << " num_devices:" << num_devices << endl;  

	_deviceId = (_deviceId < num_devices) ? _deviceId : num_devices - 1;
	cudaSetDevice(_deviceId);

	// use requested workgroup size, but we require multiple of 8
	m_workgroup_size = ((workgroup_size + 7) / 8) * 8;

	// create buffer for dag
	char* dag_ptr;
	r = cudaMalloc(&dag_ptr, _dagSize);
	if (r != cudaSuccess)
	{
		cout << "error: cudaMalloc(&dag_ptr, _dagSize): " << cudaGetErrorString(r) << endl;
		return false;
	}

	cout << "m_header:" << (void *)m_header << endl;
	// create buffer for header
	r = cudaMalloc(&m_header, 32);
	if (r != cudaSuccess)
	{
		cout << "error: cudaMalloc(&m_header, 32);" << cudaGetErrorString(r) << endl;
		return false;
	}
	cout << "m_header:" << m_header << endl;

	cudaMemcpy((void *)_dag, dag_ptr, _dagSize, cudaMemcpyHostToDevice);

	// create mining buffers
	for (unsigned i = 0; i != c_num_buffers; ++i)
	{
		r = cudaMalloc(&m_hash_buf[i], 32*c_hash_batch_size);
		if (r != cudaSuccess)
		{
			cout << "error: cudaMalloc(&m_hash_buf[i], 32*c_hash_batch_size);" << cudaGetErrorString(r) << endl;
			return false;
		}
		r = cudaMalloc(&m_search_buf[i], (c_max_search_results + 1) * sizeof(uint32_t));
		if (r != cudaSuccess)
		{
			cout << "error: cudaMalloc(&m_search_buf[i], (c_max_search_results + 1) * sizeof(uint32_t));" << cudaGetErrorString(r) << endl;
			return false;
		}
	}
	return true;
}

/******************************************
* FUNCTION: hash
*******************************************/
struct pending_batch
{
	unsigned base;
	unsigned count;
	unsigned buf;
};

void ethash_cuda_miner::hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count)
{
	cout << "hash" << endl;
	std::queue<pending_batch> pending;

	cudaMemcpy( m_header, header, 32, cudaMemcpyHostToDevice);

	unsigned buf = 0;
	for (unsigned i = 0; i < count || !pending.empty(); )
	{
		// how many this batch
		if (i < count)
		{
			unsigned const this_count = std::min(count - i, (unsigned int) c_hash_batch_size);
			unsigned const batch_count = std::max(this_count, m_workgroup_size);

			pending_batch temp_pending_batch;
			temp_pending_batch.base = i;
			temp_pending_batch.count = this_count;
			temp_pending_batch.buf = buf;

			// execute it!
			ethash_hash<<<batch_count, m_workgroup_size>>>((hash32_t*)m_hash_buf[buf],
					(const hash32_t*)m_header, (const hash128_t*)m_dag, nonce, ~0U);

			pending.push(temp_pending_batch);
			i += this_count;
			buf = (buf + 1) % c_num_buffers;
		}

		// read results
		if (i == count || pending.size() == c_num_buffers)
		{
			pending_batch const& batch = pending.front();

			// could use pinned host pointer instead, but this path isn't that important.
			cudaMemcpy(ret + batch.base*ETHASH_BYTES, m_hash_buf[batch.buf], batch.count*ETHASH_BYTES, cudaMemcpyDeviceToHost );

			pending.pop();
		}
	}
}

/******************************************
* FUNCTION: search
*******************************************/
struct pending_batch_search
{
	uint64_t start_nonce;
	unsigned buf;
};

__host__ void ethash_cuda_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{
	cudaError kerr;
	std::queue<pending_batch_search> pending;
	static uint32_t const c_zero = 0;
	uint32_t* results = (uint32_t*)malloc((1+c_max_search_results) * sizeof(uint32_t));

	// update header constant buffer
    cout << "m_header:" << (void *)m_header << endl;
	cudaMemcpy(m_header, header, 32, cudaMemcpyHostToDevice);
	kerr = cudaGetLastError();
	if (cudaSuccess != kerr)
	{
		cout << "error: cudaMemcpy hostToDevice m_header: " << cudaGetErrorString(kerr) << endl;
		return;
	}

	for (unsigned i = 0; i != c_num_buffers; ++i)
		cudaMemcpy( m_search_buf[i], &c_zero, 4, cudaMemcpyHostToDevice);
	kerr = cudaGetLastError();
	if (cudaSuccess != kerr)
	{
		cout << "error: cudaMemcpy hostToDevice m_search_buf: " << cudaGetErrorString(kerr) << endl;
		return;
	}

   
	unsigned buf = 0;

	for (uint64_t start_nonce = 0; ; start_nonce += c_search_batch_size)
	{
		// execute it!
		dim3 dimGrid(512, c_search_batch_size/512, 1);
		ethash_search<<<dimGrid, m_workgroup_size>>>((uint*)m_search_buf[buf],
				(hash32_t const*)m_header, (hash128_t const*)m_dag, start_nonce, target, ~0U);

		kerr = cudaGetLastError();
		if (cudaSuccess != kerr)
		{
			cout << "error: ethash_searchL:" << cudaGetErrorString(kerr) << endl;
		}

		pending_batch_search temp_pending_batch;
		temp_pending_batch.start_nonce = start_nonce;
		temp_pending_batch.buf = buf;

		pending.push(temp_pending_batch);
		buf = (buf + 1) % c_num_buffers;
		
		// read results
		if (pending.size() == c_num_buffers)
		{
			pending_batch_search const& batch = pending.front();

			// could use pinned host pointer instead
			cudaMemcpy( results, m_search_buf[batch.buf], (1+c_max_search_results) * sizeof(uint32_t), cudaMemcpyDeviceToHost );

			unsigned num_found = std::min<unsigned>(results[0], (uint32_t) c_max_search_results);

			uint64_t nonces[c_max_search_results];
			for (unsigned i = 0; i != num_found; ++i)
				nonces[i] = batch.start_nonce + results[i+1];

			bool exit = num_found && hook.found(nonces, num_found);
			exit |= hook.searched(batch.start_nonce, c_search_batch_size); // always report searched before exit
			if (exit) {
				break;
			}
				
			// end search prematurely due to poor performance
			//if(start_nonce == 524288) {
			//	cout << "premature exit";
			//	break;
			//}

			// reset search buffer if we're still going
			cout << "num_found " << num_found << endl;
 			if (num_found)
				cudaMemcpy(m_search_buf[batch.buf], &c_zero, 4, cudaMemcpyHostToDevice);

			pending.pop();
			//cout << "after pending.pop()" << endl;
		}
	}
	
	delete[] results;
}