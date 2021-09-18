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

void SampleDecoder::Init() const{
    return;
}

void SampleDecoder::Decode(float* d_next, float* d_nextFitKeys) const {
    int p = 256;
    int n = 32;
	dec<<<p, n>>>(d_next, d_nextFitKeys);
    return;
}
