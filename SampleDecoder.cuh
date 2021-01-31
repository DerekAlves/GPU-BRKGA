#ifndef SAMPLEDECODER_CUH
#define SAMPLEDECODER_CUH

#include "cub/cub.cuh"

//sample decoder example
class SampleDecoder {
public:
	SampleDecoder();	// Constructor
	~SampleDecoder();	// Destructor

	void deco(int blk, int thr, float* d_next, float* d_nextFitKeys, int* d_nextFitValues) const;
};

#endif