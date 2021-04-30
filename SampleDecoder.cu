/* SampleDecoder.cu
This file should be specificated by the user.

Authors
Derek N.A. Alves, 
Bruno C.S. Nogueira, 
Davi R.C Oliveira and
Ermeson C. Andrade.

Instituto de Computação, Universidade Federal de Alagoas.
Maceió, Alagoas, Brasil.
*/
 

#include "SampleDecoder.cuh"

#include "k.cuh"

SampleDecoder::SampleDecoder() { }

SampleDecoder::~SampleDecoder() { }

void SampleDecoder::deco(int blk, int thr, float* d_next, float* d_nextFitKeys, int* d_nextFitValues) const {
	dec<<<blk, thr>>>(d_next, d_nextFitKeys, d_nextFitValues);
    return;
}
