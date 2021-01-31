
#include "SampleDecoder.cuh"

#include "k.cuh"

SampleDecoder::SampleDecoder() { }

SampleDecoder::~SampleDecoder() { }

//decoder example
void SampleDecoder::deco(int blk, int thr, float* d_next, float* d_nextFitKeys, int* d_nextFitValues) const {
	dec<<<blk, thr>>>(d_next, d_nextFitKeys, d_nextFitValues);// decoder kernel
    return;
}
