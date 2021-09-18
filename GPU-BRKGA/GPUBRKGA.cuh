/* GPUBRKGA.cuh
This file countains the GPU-BRKGA class,
should not be modified.

Authors
Derek N.A. Alves, 
Bruno C.S. Nogueira, 
Davi R.C Oliveira and
Ermeson C. Andrade.

Instituto de Computação, Universidade Federal de Alagoas.
Maceió, Alagoas, Brasil.
*/
 

#ifndef GPUBRKGA_CUH
#define GPUBRKGA_CUH

#define max_t 512

#include "kernels.cuh"
#include <curand_kernel.h>
#include <curand.h>
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <exception>
#include <stdexcept>
#include "Individual.h"
#include "cuda_errorchecking.h"

template< class Decoder >
class GPUBRKGA {
public:

	/*
	Default constructor
	 Required parameters:
	 - n: number of genes in each chromosome
	 - p: number of elements in each population
	 - pe: pct of elite items into each population
	 - pm: pct of mutants introduced at each generation into the population
	 - rhoe: probability that an offspring inherits the allele of its elite parent
	 - RefDecoder: Reference to the decoder specified by user
	 - seed: Seed for the random number generator
	 Optional parameters
	 - K: number of independent populations
	 */
	GPUBRKGA(	unsigned n, unsigned p, double pe, double pm, double rhoe, const Decoder& refDecoder, int seed, bool gpu_deco,
			unsigned K = 1) throw(std::range_error);

	// destructor
	~GPUBRKGA();

	// reset all populations with new keys
	void reset();

	// evolves populations
	void evolve(unsigned generations = 1);

	// exchange top M elite individuals from populations
	void exchangeElite(unsigned M) throw(std::range_error);

	// returns data from gpu
	 std::vector<std::vector<Individual*>> getPopulations();
	 Individual getBestIndividual();
	 void cpyHost();
	// initializes the GPU-BRKGA
	void initializeGPU();
	

	// Return copies to the internal parameters:
	unsigned getN() const;
	unsigned getP() const;
	unsigned getPe() const;
	unsigned getPm() const;
	unsigned getPo() const;
	double getRhoe() const;
	unsigned getK() const;
	unsigned getThr() const;
	float* getHostPopulations() const;
	float* getHostFitnessKeys() const;
	int* getHostFitnessValues() const;

private:

	// parameters:
	const unsigned n;	// number of genes in the chromosome
	const unsigned p;	// number of elements in the population
	const unsigned pe;	// number of elite items in the population
	const unsigned pm;	// number of mutants introduced at each generation into the population
	const double rhoe;	// probability that an offspring inherits the allele of its elite parent
	const int seed;		// seed for the rng
	const Decoder& refDecoder; // reference to decoder
	const unsigned K; // number of independent populations
	const bool gpu_deco; // flag for GPU or CPU decode

	unsigned thr;	// items to be computed for thread due to restrictions in thread number per block

	// Host data pointers
	float* h_populations;
	float* h_fitnessKeys;
	int* h_fitnessValues;

	float* temp_pop;
	float* temp_keys;

	// Device data pointers
	//Populations curr and prev
	float* d_current;
	float* d_previous;
	// Fitness of each individual from population curr and prev
	float* d_currFitnessKeys;
	int* d_currFitnessValues;
	float* d_prevFitnessKeys;
	int* d_prevFitnessValues;
	//

	// curand rng states
	curandState* d_crossStates;
	curandState* d_mateStates;

	// Local operations:
	void evolution();
	bool isRepeated(const std::vector< double >& chrA, const std::vector< double >& chrB) const;
	unsigned getInd(unsigned _k, unsigned i, bool t) const;
	unsigned getOffset(int _k, bool _pop) const;
};

template< class Decoder >
void GPUBRKGA< Decoder >::cpyHost()
{
	if(h_populations == NULL)
	{
		h_populations 	= (float*)malloc((K*p*n)*sizeof(float));
		h_fitnessKeys 	= (float*)malloc((K*p)*sizeof(float));
		h_fitnessValues = (int*)malloc((K*p)*sizeof(int));
	}
	
	cudaMemcpy(h_populations, d_current, K*p*n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fitnessKeys, d_currFitnessKeys, K*p*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fitnessValues, d_currFitnessValues, K*p*sizeof(int), cudaMemcpyDeviceToHost);
}

template< class Decoder >
GPUBRKGA< Decoder >::GPUBRKGA(unsigned _n, unsigned _p, double _pe, double _pm, double _rhoe, const Decoder& _refDecoder, int _seed, bool _gpu_deco,
		unsigned _K) throw(std::range_error) :
		n(_n), p(_p), pe(unsigned(_pe * p)), pm(unsigned(_pm * p)), rhoe(_rhoe), refDecoder(_refDecoder), seed(_seed), gpu_deco(_gpu_deco),
		K(_K){
	// Error check:
	using std::range_error;
	if(n == 0) { throw range_error("Chromosome size equals zero."); }
	if(p == 0) { throw range_error("Population size equals zero."); }
	if(pe == 0) { throw range_error("Elite-set size equals zero."); }
	if(pe > p) { throw range_error("Elite-set size greater than population size (pe > p)."); }
	if(pm > p) { throw range_error("Mutant-set size (pm) greater than population size (p)."); }
	if(pe + pm > p) { throw range_error("elite + mutant sets greater than population size (p)."); }
	if(K == 0) { throw range_error("Number of parallel populations cannot be zero."); }

	thr = min(n, max_t);
	// Host data alloc
	h_populations 	= NULL;
	h_fitnessKeys 	= NULL;
	h_fitnessValues = NULL;
	temp_pop = NULL;
	temp_keys = NULL;

	// Device data alloc
	cudaMalloc((void**)&d_current, (K*p*n) * sizeof(float));
	cudaMalloc((void**)&d_previous, (K*p*n) * sizeof(float));

	cudaMalloc((void**)&d_currFitnessKeys, (K*p) * sizeof(float));
	cudaMalloc((void**)&d_currFitnessValues, (K*p) * sizeof(int));
	cudaMalloc((void**)&d_prevFitnessKeys, (K*p) * sizeof(float));
	cudaMalloc((void**)&d_prevFitnessValues, (K*p) * sizeof(int));
	
	// RNG states
	cudaMalloc((void**)&d_crossStates, (p*thr) * sizeof(curandState));
	cudaMalloc((void**)&d_mateStates, 2 * p * sizeof(curandState));
	
	//init_genrand(seed);
	setup_kernel<<<p, thr>>>(d_crossStates, d_mateStates, seed);

	refDecoder.Init();

	initializeGPU();
	
}

template< class Decoder >
GPUBRKGA< Decoder >::~GPUBRKGA() {
	cudaFree(d_current);
	cudaFree(d_previous);
	cudaFree(d_currFitnessKeys);
	cudaFree(d_currFitnessValues);
	cudaFree(d_prevFitnessKeys);
	cudaFree(d_prevFitnessValues);
	cudaFree(d_crossStates);
	cudaFree(d_mateStates);

	if(h_populations != NULL){
		free(h_populations);
		free(h_fitnessKeys);
		free(h_fitnessValues);
	}

	if(temp_pop != NULL){
		free(temp_pop);
		free(temp_keys);
	}

}

template< class Decoder >
Individual GPUBRKGA< Decoder >::getBestIndividual() {
	int h_bk;
	int* d_bk;
	
	cudaMalloc((void**)&d_bk, sizeof(int));
	bestK<<<1,1>>>(d_currFitnessKeys, K, p, d_bk);
	cudaMemcpy(&h_bk, d_bk, sizeof(float), cudaMemcpyDeviceToHost);
	float* al = (float*)malloc(n*sizeof(float));
	std::pair<float, int> ft;

	cudaMemcpy(&ft.first, d_currFitnessKeys + getOffset(h_bk, false), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&ft.second, d_currFitnessValues + getOffset(h_bk, false), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(al, d_current + getInd(h_bk, ft.second, true), n * sizeof(float), cudaMemcpyDeviceToHost);

	Individual bInd(al, ft);

	return bInd;
}

template< class Decoder >
void GPUBRKGA< Decoder >::reset() {
	initializeGPU();
}

template< class Decoder >
void GPUBRKGA< Decoder >::evolve(unsigned generations) {
	#ifdef RANGECHECK
		if(generations == 0) { throw std::range_error("Cannot evolve for 0 generations."); }
	#endif
	for(unsigned i = 0; i < generations; ++i) {
		evolution();
	// Update (prev = curr; curr = prev == next)
	std::swap(d_current, d_previous);
	}
}

template< class Decoder >
void GPUBRKGA< Decoder >::exchangeElite(unsigned M) throw(std::range_error) {
	#ifdef RANGECHECK
		if(M == 0 || M >= p) { throw std::range_error("M cannot be zero or >= p."); }
	#endif

	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	exchange_te<<<K,1>>>(d_current, d_currFitnessKeys, d_currFitnessValues, K, p, n, M);

	for(int i = 0; i < K; i++)
	{
		if(d_temp_storage == NULL)
		{
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
				d_currFitnessKeys + (i*p), d_prevFitnessKeys + (i*p), d_currFitnessValues + (i*p), d_prevFitnessValues + (i*p), p);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}
		// Run sorting operation
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
			d_currFitnessKeys + (i*p), d_prevFitnessKeys + (i*p), d_currFitnessValues + (i*p), d_prevFitnessValues + (i*p), p);
	}
	std::swap(d_currFitnessKeys, d_prevFitnessKeys);
	std::swap(d_currFitnessValues, d_prevFitnessValues);

	//cudaFree(d_temp_storage);
}

template< class Decoder >
std::vector<std::vector<Individual*>> GPUBRKGA< Decoder>::getPopulations(){

	if(h_populations == NULL)
	{
		h_populations 	= (float*)malloc((K*p*n)*sizeof(float));
		h_fitnessKeys 	= (float*)malloc((K*p)*sizeof(float));
		h_fitnessValues = (int*)malloc((K*p)*sizeof(int));
	}
	
	cudaMemcpy(h_populations, d_current, K*p*n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fitnessKeys, d_currFitnessKeys, K*p*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fitnessValues, d_currFitnessValues, K*p*sizeof(int), cudaMemcpyDeviceToHost);
	
	int idx_chr, idx_f;
	std::pair<float, int> ft;
	
	std::vector<std::vector<Individual*>> pops;
	
	for(int _k = 0; _k < K; _k++)
	{
		std::vector<Individual*> aux;
		for(int _p = 0; _p < p; _p++)
		{
			idx_f = getInd(_k, _p, false);
			ft.first = h_fitnessKeys[idx_f];
			ft.second = h_fitnessValues[idx_f];
			aux.push_back(new Individual(h_populations + idx_chr, ft));
		}
		pops.push_back(aux);

	}		
	return pops;
}

template< class Decoder >
inline void GPUBRKGA< Decoder >::initializeGPU()
{
	int offp, offf;
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	for(int i = 0; i < K; i++)
	{
		offp = getOffset(i, true);
		offf = getOffset(i, false);

		gpuInit<<<p, thr>>>(n, d_current + offp, d_currFitnessValues + offf, d_crossStates);

		if(gpu_deco) refDecoder.Decode(d_current, d_currFitnessKeys);
		else{
			if(temp_pop == NULL){
				temp_pop = (float*)malloc(p*n*sizeof(float));
				temp_keys = (float*)malloc(p*sizeof(float));
			}
			
			cudaMemcpy(temp_pop, d_current + offp, p*n*sizeof(float), cudaMemcpyDeviceToHost);
			refDecoder.Decode(temp_pop, temp_keys);
			cudaMemcpy(d_currFitnessKeys + offf, temp_keys, p*sizeof(float), cudaMemcpyHostToDevice);
		}

		if(d_temp_storage == NULL)
		{
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
				d_currFitnessKeys + (i*p), d_prevFitnessKeys + (i*p), d_currFitnessValues + (i*p), d_prevFitnessValues + (i*p), p);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}
		// Run sorting operation
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
			d_currFitnessKeys + (i*p), d_prevFitnessKeys + (i*p), d_currFitnessValues + (i*p), d_prevFitnessValues + (i*p), p);

	}
	std::swap(d_currFitnessKeys, d_prevFitnessKeys);
	std::swap(d_currFitnessValues, d_prevFitnessValues);	
}

template< class Decoder >
inline void GPUBRKGA< Decoder >::evolution() {
	
	unsigned offp, offf;
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	for(int _k = 0; _k < K; _k++)
	{
		offp = getOffset(_k, true);
		offf = getOffset(_k, false);

		offspring<<<p, n>>>(d_current + offp, d_previous + offp, d_currFitnessValues + offf, d_prevFitnessValues + offf,
			p, pe, pm, rhoe, n, d_crossStates, d_mateStates);
		
		if(gpu_deco) refDecoder.Decode(d_previous + offp, d_prevFitnessKeys + offf);
		else{
			if(temp_pop == NULL){
				temp_pop = (float*)malloc(p*n*sizeof(float));
				temp_keys = (float*)malloc(p*sizeof(float));
			}
			
			cudaMemcpy(temp_pop, d_previous + offp, p*n*sizeof(float), cudaMemcpyDeviceToHost);
			refDecoder.Decode(temp_pop, temp_keys);
			cudaMemcpy(d_prevFitnessKeys + offf, temp_keys, p*sizeof(float), cudaMemcpyHostToDevice);
		}	 

		// RADIX-SORT
		// sorting fitness next
		if(d_temp_storage == NULL)
		{
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
				d_prevFitnessKeys, d_currFitnessKeys, d_prevFitnessValues, d_currFitnessValues, p);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}
		
		// Run sorting operation
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
			d_prevFitnessKeys, d_currFitnessKeys, d_prevFitnessValues, d_currFitnessValues, p);
	}
}

template< class Decoder >
float* GPUBRKGA< Decoder >::getHostPopulations() const { return h_populations; }

template< class Decoder >
float* GPUBRKGA< Decoder >::getHostFitnessKeys() const { return h_fitnessKeys; }

template< class Decoder >
int* GPUBRKGA< Decoder >::getHostFitnessValues() const { return h_fitnessValues; }

template< class Decoder >
unsigned GPUBRKGA< Decoder >::getN() const { return n; }

template< class Decoder >
unsigned GPUBRKGA< Decoder >::getP() const { return p; }

template< class Decoder >
unsigned GPUBRKGA< Decoder >::getPe() const { return pe; }

template< class Decoder >
unsigned GPUBRKGA< Decoder >::getPm() const { return pm; }

template< class Decoder >
unsigned GPUBRKGA< Decoder >::getPo() const { return p - pe - pm; }

template< class Decoder >
double GPUBRKGA< Decoder >::getRhoe() const { return rhoe; }

template< class Decoder >
unsigned GPUBRKGA< Decoder >::getK() const { return K; }

template< class Decoder >
unsigned GPUBRKGA< Decoder >::getThr() const { return thr; }

template< class Decoder >
unsigned GPUBRKGA< Decoder >::getOffset(int _k, bool _pop) const { 
	if(_pop)return _k*p*n; // true, return chromosomes of population k offset
	else return _k*p; // else return fitness of population k offset
}

template< class Decoder > // get index to chromosome/fitness i of population k
unsigned GPUBRKGA< Decoder >::getInd(unsigned _k, unsigned i, bool t) const { 
	if(t) return _k*p*n + i*n;
	else return _k*p + i;
}

#endif