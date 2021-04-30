/* SampleDecoder.cuh
This file should be specificated by the user.

Authors
Derek N.A. Alves, 
Bruno C.S. Nogueira, 
Davi R.C Oliveira and
Ermeson C. Andrade.

Instituto de Computação, Universidade Federal de Alagoas.
Maceió, Alagoas, Brasil.
*/

#ifndef SAMPLEDECODER_CUH
#define SAMPLEDECODER_CUH

#include "cub/cub.cuh"
//#include "k.cuh"

class SampleDecoder {
public:
	SampleDecoder();	// Constructor
	~SampleDecoder();	// Destructor

	void deco(int blk, int thr, float* d_next, float* d_nextFitKeys, int* d_nextFitValues) const;
};

#endif