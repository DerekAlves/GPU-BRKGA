/* kernels.cuh
This file contains auxiliary kernels and functions for
paralel execution. Should not be modified.

Authors
Derek N.A. Alves, 
Bruno C.S. Nogueira, 
Davi R.C Oliveira and
Ermeson C. Andrade.

Instituto de Computação, Universidade Federal de Alagoas.
Maceió, Alagoas, Brasil.
*/
 

#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "cub/cub.cuh"
#include <curand_kernel.h>


//CURAND RNG

__global__ void setup_kernel(curandState *state, curandState *state2, int seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &state[idx]);
	if(threadIdx.x == 0 || threadIdx.x == 1)
	{
		idx = threadIdx.x + blockIdx.x * 2;
		curand_init(seed, idx, 0, &state2[idx]);
	}
}

__device__ int RNG_int(unsigned n, int mod) // returns number % mod!
{ return n % mod; }

__device__ float RNG_real(unsigned n) // returns [0, 1) interval
{ return n * (1.0/4294967296.0); }

__global__ void gpuInit(int n, float* d_pop, int* d_val, curandState* d_crossStates)
{
	unsigned int idx = threadIdx.x + blockIdx.x * n;
	unsigned int bidx =  idx;
	unsigned int auxidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int offset = blockDim.x;	

	for(; idx < bidx + n; idx+=offset)
            d_pop[idx] = RNG_real(curand(&d_crossStates[auxidx]));

	if(!threadIdx.x)
		d_val[blockIdx.x] = blockIdx.x;
}

//KERNELS

__global__ void bestK(float* keys, unsigned K, unsigned p, int* bk)
{	
	if(bk != NULL)
	{
		float best = keys[0];
		*bk = 0;
		for(int i = 1; i < K; i++)
		{
			if( best > keys[i*p])
			{
				*bk = i;
				best = keys[i*p];
			}
		}	
	}
	return;
}

__global__ void offspring(float* d_current, float* d_next, int* d_currFitValues,
	int* d_nextFitValues, int P, int PE, int PM, float rhoe, unsigned int n, 
	curandState* d_crossStates, curandState* d_mateStates) 
{
	
	unsigned int idx = threadIdx.x + blockIdx.x * n;
	unsigned int bidx =  idx;
	unsigned int auxidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int offset = blockDim.x;	

    if(blockIdx.x < PE)
    {
		
        unsigned int pblk = d_currFitValues[blockIdx.x];
        unsigned int pidx = threadIdx.x + pblk*n;
        for(; idx < bidx + n; idx+=offset, pidx+=offset)
            d_next[idx] = d_current[pidx];
		
    }
    else if(blockIdx.x < P - PM)
    {
        __shared__ unsigned int eliteParent;
		__shared__ unsigned int nonEliteParent;

		if(!threadIdx.x)
		{
			eliteParent = RNG_int(curand(&d_mateStates[blockIdx.x * 2]), PE);
			nonEliteParent = PE + RNG_int(curand(&d_mateStates[(blockIdx.x + 1) * 2 - 1]), P-PE);
		}
        __syncthreads();
        unsigned int epidx = threadIdx.x + eliteParent*n;
        unsigned int pidx = threadIdx.x + nonEliteParent*n;
        unsigned int sp;
        float prob;
        for(; idx < bidx + n; idx+=offset, pidx+=offset, epidx+=offset){
            prob = RNG_real(curand(&d_crossStates[auxidx]));
            sp = prob < rhoe ? epidx : pidx;
            d_next[idx] = d_current[sp];
        }
    }
    else if(blockIdx.x < P)
    {
        for(; idx < bidx + n; idx+=offset)
            d_next[idx] = RNG_real(curand(&d_crossStates[auxidx]));
    }

    if(!threadIdx.x)
		d_nextFitValues[blockIdx.x] = blockIdx.x;
}

__global__ void exchange_te(float* d_populations, float* d_fitKeys, int* d_fitValues, int k, int p, int n, int top)
{
	int idx = blockIdx.x;
	int dest = p-1;

	int pop1_idx, pop2_idx;
	int fit1_idx, fit2_idx;

	for(unsigned int i = 0; i < k; i++)
	{
		if(i == idx) continue;
		
		for(unsigned int j = 0; j < top; j++)
		{ 
			fit1_idx = idx*p + dest;
			fit2_idx = i*p + j;
			pop1_idx = idx*p*n + d_fitValues[fit1_idx]*n;
			pop2_idx = i*p*n + d_fitValues[fit2_idx]*n;
			memcpy(d_populations + pop1_idx, d_populations + pop2_idx, sizeof(int)*n);//copy aleles
			d_fitKeys[fit1_idx] = d_fitKeys[fit2_idx]; // copy fitness
			dest--;
		}
	}
}
#endif