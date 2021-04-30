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

template< class Decoder >
class GPUBRKGA {
public:
	/*
	 * Default constructor
	 * Required hyperparameters:
	 * - n: number of genes in each chromosome
	 * - p: number of elements in each population
	 * - pe: pct of elite items into each population
	 * - pm: pct of mutants introduced at each generation into the population
	 * - rhoe: probability that an offspring inherits the allele of its elite parent
	 *
	 * Optional parameters:
	 * - K: number of independent Populations
	 * - MAX_THREADS: number of threads to perform parallel decoding
	 *                WARNING: Decoder::decode() MUST be thread-safe; safe if implemented as
	 *                + double Decoder::decode(std::vector< double >& chromosome) const
	 */
	GPUBRKGA(	unsigned n, unsigned p, double pe, double pm, double rhoe, const Decoder& refDecoder, int seed, unsigned it_perthr,
			unsigned K = 1) throw(std::range_error);

	/**
	 * Destructor
	 */
	~GPUBRKGA();

	/**
	 * Resets all populations with brand new keys
	 */
	void reset();

	/**
	 * Evolve the current populations following the guidelines of BRKGAs
	 * @param generations number of generations (must be even and nonzero)
	 * @param J interval to exchange elite chromosomes (must be even; 0 ==> no synchronization)
	 * @param M number of elite chromosomes to select from each population in order to exchange
	 */
	void evolve(unsigned generations = 1);

	/**
	 * Exchange elite-solutions between the populations
	 * @param M number of elite chromosomes to select from each population
	 */
	void exchangeElite(unsigned M) throw(std::range_error);

	/**
	 * Returns the current population
	 */
	//const Population& getPopulation(unsigned k = 0) const;

	/**
	 * Returns the chromosome with best fitness so far among all populations
	 */
	//const void getBestChromosome(float *ptr) const;

	/**
	 * Returns the best fitness found so far among all populations
	 */
	//float getBestFitness() const;
	void cpyHost();
	void initializeGPU();
	Individual getBestIndividual();

	// Return copies to the internal parameters:
	unsigned getN() const;
	unsigned getP() const;
	unsigned getPe() const;
	unsigned getPm() const;
	unsigned getPo() const;
	double getRhoe() const;
	unsigned getK() const;
	float* getPopulations() const;
	float* getFitnessKeys() const;
	int* getFitnessValues() const;
	unsigned getOffset(int _k, bool _pop) const;
	unsigned getChromosome(unsigned _k, unsigned i) const;
	std::vector<std::vector<Individual*>> getDeviceInfo();
	unsigned getInd(unsigned _k, unsigned i, bool t) const;

private:

	// Hyperparameters:
	const unsigned n;	// number of genes in the chromosome
	const unsigned p;	// number of elements in the population
	const unsigned pe;	// number of elite items in the population
	const unsigned pm;	// number of mutants introduced at each generation into the population
	const double rhoe;	// probability that an offspring inherits the allele of its elite parent
	const int seed;		// seed for the rng
	const unsigned ipt;	// items to be computed for thread due to restrictions in thread number per block.

	const Decoder& refDecoder;

	// Parallel populations parameters:
	const unsigned K;				// number of independent parallel populations

	// Host data pointers
	float* h_populations;
	float* h_fitnessKeys;
	int* h_fitnessValues;

	// Device data pointers
	//Populations curr and prev
	float* d_current;
	float* d_previous;
	// Fitness of each individual from population curr and prev
	float* d_currFitnessKeys;
	int* d_currFitnessValues;
	float* d_prevFitnessKeys;
	int* d_prevFitnessValues;

	//RNG STATES
	curandState* d_crossStates;
	curandState* d_mateStates;

	// Local operations:
	void evolution();
	bool isRepeated(const std::vector< double >& chrA, const std::vector< double >& chrB) const;
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
GPUBRKGA< Decoder >::GPUBRKGA(unsigned _n, unsigned _p, double _pe, double _pm, double _rhoe, const Decoder& _refDecoder, int _seed, unsigned it_perthr,
		unsigned _K) throw(std::range_error) :
		n(_n), p(_p), pe(unsigned(_pe * p)), pm(unsigned(_pm * p)), rhoe(_rhoe), refDecoder(_refDecoder), seed(_seed), ipt(it_perthr),
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


	// Host data alloc
	h_populations 	= NULL;
	h_fitnessKeys 	= NULL;
	h_fitnessValues = NULL;

	// Device data alloc
	cudaMalloc((void**)&d_current, (K*p*n) * sizeof(float));
	cudaMalloc((void**)&d_previous, (K*p*n) * sizeof(float));

	cudaMalloc((void**)&d_currFitnessKeys, (K*p) * sizeof(float));
	cudaMalloc((void**)&d_currFitnessValues, (K*p) * sizeof(int));
	cudaMalloc((void**)&d_prevFitnessKeys, (K*p) * sizeof(float));
	cudaMalloc((void**)&d_prevFitnessValues, (K*p) * sizeof(int));
	
	// RNG states
	cudaMalloc((void**)&d_crossStates, (p*n) * sizeof(curandState));
	cudaMalloc((void**)&d_mateStates, 2 * p * sizeof(curandState));
	
	//init_genrand(seed);
	setup_kernel<<<p, n>>>(d_crossStates, d_mateStates, seed);

	initializeGPU();
	
}

template< class Decoder >
GPUBRKGA< Decoder >::~GPUBRKGA() {
	for(unsigned i = 0; i < K; ++i) {  }
}


/*const Population& BRKGA<>::getPopulation(unsigned k) const {
	#ifdef RANGECHECK
		if(k >= K) { throw std::range_error("Invalid population identifier."); }
	#endif
	return ();
}*/

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

		gpuInit<<<p, n>>>(d_current + offp, d_currFitnessValues + offf, d_crossStates, ipt);
		refDecoder.deco(K*p, n, d_current, d_currFitnessKeys, d_currFitnessValues);

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
	// We now will set every chromosome of

	unsigned offp, offf;
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	for(int _k = 0; _k < K; _k++)
	{
		offp = getOffset(_k, true);
		offf = getOffset(_k, false);

		offspring<<<p, n>>>(d_current + offp, d_previous + offp, d_currFitnessKeys + offf, d_currFitnessValues + offf, d_prevFitnessKeys + offf, d_prevFitnessValues + offf,
			p, pe, pm, rhoe, ipt, d_crossStates, d_mateStates);
		
		refDecoder.deco(p, n, d_previous + offp, d_prevFitnessKeys + offf, d_prevFitnessValues + offf);

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
float* GPUBRKGA< Decoder >::getPopulations() const { return h_populations; }

template< class Decoder >
float* GPUBRKGA< Decoder >::getFitnessKeys() const { return h_fitnessKeys; }

template< class Decoder >
int* GPUBRKGA< Decoder >::getFitnessValues() const { return h_fitnessValues; }

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
unsigned GPUBRKGA< Decoder >::getOffset(int _k, bool _pop) const { 
	if(_pop)return _k*p*n; // true, return chromosomes of population k offset
	else return _k*p; // else return fitness of population k offset
}

template< class Decoder > // get gpu pointer to chromosome i of population k in current
unsigned GPUBRKGA< Decoder >::getChromosome(unsigned _k, unsigned i) const { 
	return _k*p*n + i*n;
}

template< class Decoder >
std::vector<std::vector<Individual*>> GPUBRKGA< Decoder>::getDeviceInfo(){

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

template< class Decoder > // get index to chromosome/fitness i of population k
unsigned GPUBRKGA< Decoder >::getInd(unsigned _k, unsigned i, bool t) const { 
	if(t) return _k*p*n + i*n;
	else return _k*p + i;
}

#endif