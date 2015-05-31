#pragma once

#include <time.h>
#include <libethash/ethash.h>

class ethash_cuda_miner
{
public:
	struct search_hook
	{
		// reports progress, return true to abort
		virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
		virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
	};

public:
	ethash_cuda_miner();
	~ethash_cuda_miner();

	static unsigned get_num_platforms();
	static unsigned get_num_devices(unsigned _platformId = 0);
	static std::string platform_info(unsigned _platformId = 0, unsigned _deviceId = 0);

	//bool init(ethash_params const& params, ethash_h256_t const *seed, unsigned workgroup_size = 64);
    bool init(uint8_t const* _dag, uint64_t _dagSize, unsigned workgroup_size = 64, unsigned _platformId = 0, unsigned _deviceId = 0);

	void finish();
	void hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count);
	void search(uint8_t const* header, uint64_t target, search_hook& hook);

private:
	//static unsigned const c_max_search_results = 63;
	//static unsigned const c_num_buffers = 2;
	//static unsigned const c_hash_batch_size = 512;
	//static unsigned const c_search_batch_size = 512*256;

	enum { c_max_search_results = 63, c_num_buffers = 2, c_hash_batch_size = 1024, c_search_batch_size = 1024*256 };
	char* m_dag;
	char* m_header;
	char* m_hash_buf[c_num_buffers];
	char* m_search_buf[c_num_buffers];
	unsigned m_workgroup_size;
};
