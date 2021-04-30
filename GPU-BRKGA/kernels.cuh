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

//UTILS
void printFitness(float* keys, int* values, int sz)
{
	for(int i = 0; i < sz; i++)
		printf("Fitness: %9.3f", keys[i]);

}

void printPopulation(float* population, float* keys, int* values, int sz, int n)
{
	for(int i = 0; i < sz; i++)
	{
		if(i % n == 0)printf("\n\nIndividuo: %d key %.2f value %d\n", i / n, keys[i/n], values[i/n] );
		printf("%.2f ", population[i]);
	}
	printf("\n---------------------------------------------------------\n");
}

void printPopulation2(float* population, int sz, int n)
{
	for(int i = 0; i < sz; i++)
	{
		if(i % n == 0)printf("\n\nIndividuo: %d\n", i / n );
		printf("%.2f ", population[i]);
	}
	printf("\n---------------------------------------------------------\n");
}

//############################################ CURAND RNG

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

__global__ void gpuInit(float* d_pop, int* d_val, curandState* d_crossStates, int ipt)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int loc_idx = idx * ipt;

	for(unsigned int i = 0; i < ipt; i++)
		d_pop[loc_idx + i] = RNG_real(curand(&d_crossStates[idx]));

	if(!threadIdx.x)
		d_val[blockIdx.x] = blockIdx.x;
}

// AUX KERNELS

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

//TESTE
__global__ void offspring(float* d_current, float* d_next, float* d_currFitKeys, int* d_currFitValues,
	float* d_nextFitKeys, int* d_nextFitValues, int P, int PE, int PM, float rhoe, unsigned int ipt, 
	curandState* d_crossStates, curandState* d_mateStates) 
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int loc_idx = idx * ipt;
	//printf("loc_idx %d\n", loc_idx);

	if(blockIdx.x < PE)
	{
		//COPY ELITE INDIVIDUALS
		for(unsigned int i = 0; i < ipt; i++)
			d_next[(loc_idx + i)] = d_current[d_currFitValues[blockIdx.x]*(blockDim.x * ipt) + threadIdx.x + i];
	}
	else if( blockIdx.x < P-PM)//GENERATE OFFSPRING
	{
		__shared__ unsigned int eliteParent;
		__shared__ unsigned int nonEliteParent;

		if(!threadIdx.x)
		{
			eliteParent = RNG_int(curand(&d_mateStates[blockIdx.x * 2]), PE);
			nonEliteParent = PE + RNG_int(curand(&d_mateStates[(blockIdx.x + 1) * 2 - 1]), P-PE);
		}
		
		__syncthreads();
		for(unsigned int i = 0; i < ipt; i++)
		{
			float prob = RNG_real(curand(&d_crossStates[idx]));
			int sp = prob < rhoe ? eliteParent : nonEliteParent;
			d_next[loc_idx + i] = d_current[d_currFitValues[sp] * (blockDim.x * ipt) + threadIdx.x + i];
		}
	}
	else if(blockIdx.x < P)//INSERT PM MUTANTS
		for(unsigned int i = 0; i < ipt; i++)
		{
			d_next[loc_idx + i] = RNG_real(curand(&d_crossStates[idx]));
		}
	if(!threadIdx.x)
		d_nextFitValues[blockIdx.x] = blockIdx.x;

}

//Copies n aleles from individual 2 to 1
__device__ void cpy(float* pop1, float* pop2, int n)
{
	for(int i = 0; i < n; i++)
		pop1[i] = pop2[i];
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
			cpy(d_populations + pop1_idx, d_populations + pop2_idx, n);
			d_fitKeys[fit1_idx] = d_fitKeys[fit2_idx]; // copies fitness
			dest--;
		}
	}
}
#endif