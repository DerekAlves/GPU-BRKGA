/* BRKGA.CUH
Derek N.A. Alves, 
Bruno C.S. Nogueira, 
Davi R.C Oliveira and
Ermeson C. Andrade.

Instituto de Computação, Universidade Federal de Alagoas.
Maceió, Alagoas, Brasil.

Based on brkgaAPI from Rodrigo F. Toso and Mauricio G.C. Resende.
*/

#ifndef BRKGA_CUH
#define BRKGA_CUH

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
class BRKGA {
public:
	//Constructor
	BRKGA(	unsigned n, unsigned p, double pe, double pm, double rhoe, const Decoder& refDecoder, int seed, unsigned it_perthr,
		unsigned thr, bool gpu_deco, unsigned K = 1) throw(std::range_error);
	//Destructor	
	~BRKGA();

	//Reset all populations with new random keys
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
	std::vector<std::vector<Individual>> getPopulations() const;
	float* getFitnessKeys() const;
	int* getFitnessValues() const;
	unsigned getOffset(int _k, bool _pop) const;
	unsigned getChromosome(unsigned _k, unsigned i) const;

private:

	// Hyperparameters:
	const unsigned n;	// number of genes in the chromosome
	const unsigned p;	// number of elements in the population
	const unsigned pe;	// number of elite items in the population
	const unsigned pm;	// number of mutants introduced at each generation into the population
	const double rhoe;	// probability that an offspring inherits the allele of its elite parent
	const int seed;		// seed for the rng
	const unsigned ipt;	// items to be computed for thread due to restrictions in thread number per block.
	const unsigned thr;
	const bool gpuDecode;

	const Decoder& refDecoder;

	// Parallel populations parameters:
	const unsigned K;	// number of independent parallel populations

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
	void initialize(const unsigned i);		
	void evolution();
	bool isRepeated(const std::vector< double >& chrA, const std::vector< double >& chrB) const;
};

template< class Decoder >
void BRKGA< Decoder >::cpyHost()
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
BRKGA< Decoder >::BRKGA(unsigned _n, unsigned _p, double _pe, double _pm, double _rhoe, const Decoder& _refDecoder, int _seed, unsigned it_perthr,
		unsigned _thr, bool gpu_deco, unsigned _K) throw(std::range_error) :
		n(_n), p(_p), pe(unsigned(_pe * p)), pm(unsigned(_pm * p)), rhoe(_rhoe), refDecoder(_refDecoder), seed(_seed), ipt(it_perthr),
		thr(_thr), gpuDecode(gpu_deco), K(_K){
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
	
	setup_kernel<<<p, thr>>>(d_crossStates, d_mateStates, seed);

	initializeGPU();
	
}

template< class Decoder >
BRKGA< Decoder >::~BRKGA() {
	for(unsigned i = 0; i < K; ++i) {  }
}

template< class Decoder >
Individual BRKGA< Decoder >::getBestIndividual() {
	int h_bk;
	int* d_bk;
	
	cudaMalloc((void**)&d_bk, sizeof(int));
	bestK<<<1,1>>>(d_currFitnessKeys, K, p, d_bk);
	cudaMemcpy(&h_bk, d_bk, sizeof(float), cudaMemcpyDeviceToHost);
	float* al = (float*)malloc(n*sizeof(float));
	std::pair<float, int> ft;

	cudaMemcpy(&ft.first, d_currFitnessKeys + getOffset(h_bk, false), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&ft.second, d_currFitnessValues + getOffset(h_bk, false), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(al, d_current + getChromosome(h_bk, ft.second), n * sizeof(float), cudaMemcpyDeviceToHost);

	Individual bInd(al, ft);

	return bInd;
}

template< class Decoder >
void BRKGA< Decoder >::reset() {
	initializeGPU();
}

template< class Decoder >
void BRKGA< Decoder >::evolve(unsigned generations) {
	#ifdef RANGECHECK
		if(generations == 0) { throw std::range_error("Cannot evolve for 0 generations."); }
	#endif
	for(unsigned i = 0; i < generations; ++i) {
		evolution();
	// Update (prev = curr; curr = prev == next)
	std::swap(d_current, d_previous);
	}
}

/* INFO ###############
	/*#ifdef RANGECHECK
		if(M == 0 || M >= p) { throw std::range_error("M cannot be zero or >= p."); }
	#endif

	for(unsigned i = 0; i < K; ++i) {
		// Population i will receive some elite members from each Population j below:
		unsigned dest = p - 1;	// Last chromosome of i (will be updated below)
		for(unsigned j = 0; j < K; ++j) {
			if(j == i) { continue; }

			// Copy the M best of Population j into Population i:
			for(unsigned m = 0; m < M; ++m) {
				// Copy the m-th best of Population j into the 'dest'-th position of Population i:
				const std::vector< double >& bestOfJ = current[j]->getChromosome(m);

				std::copy(bestOfJ.begin(), bestOfJ.end(), current[i]->getChromosome(dest).begin());

				current[i]->fitness[dest].first = current[j]->fitness[m].first;

				--dest;
			}
		}
	}

	for(int j = 0; j < int(K); ++j) { current[j]->sortFitness(); }*/

//IMPLEMENTAR / TESTAR COM MULTIPLAS POPULAÇÕES
template< class Decoder >
void BRKGA< Decoder >::exchangeElite(unsigned M) throw(std::range_error) {

}


template< class Decoder >
inline void BRKGA< Decoder >::initializeGPU()
{
	int offp, offf;
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	for(int _k = 0; _k < K; _k++)
	{
		offp = getOffset(_k, true);
		offf = getOffset(_k, false);

		gpuInit<<<p, thr>>>(d_current + offp, d_currFitnessValues + offf, d_crossStates, ipt);
		refDecoder.deco(p, thr, d_current + offp, d_currFitnessKeys + offf, d_currFitnessValues + offf);

		if(d_temp_storage == NULL)
		{
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
				d_currFitnessKeys + offf, d_prevFitnessKeys + offf, d_currFitnessValues + offf, d_prevFitnessValues + offf, p);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}
		// Run sorting operation
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
			d_currFitnessKeys + offf, d_prevFitnessKeys + offf, d_currFitnessValues + offf, d_prevFitnessValues + offf, p);

	}
	std::swap(d_currFitnessKeys, d_prevFitnessKeys);
	std::swap(d_currFitnessValues, d_prevFitnessValues);	
}

template< class Decoder >
inline void BRKGA< Decoder >::evolution() {
	// We now will set every chromosome of

	unsigned offp, offf;
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	for(int _k = 0; _k < K; _k++)
	{
		offp = getOffset(_k, true);
		offf = getOffset(_k, false);

		offspring<<<p, thr>>>(d_current + offp, d_previous + offp, d_currFitnessValues + offf, d_prevFitnessValues + offf,
			p, pe, pm, rhoe, n, d_crossStates, d_mateStates);
		
		refDecoder.deco(p, thr, d_previous + offp, d_prevFitnessKeys + offf, d_prevFitnessValues + offf);

		// RADIX-SORT
		// sorting fitness next
		if(d_temp_storage == NULL)
		{
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
				d_prevFitnessKeys + offf, d_currFitnessKeys + offf, d_prevFitnessValues + offf, d_currFitnessValues + offf, p);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}
		
		// Run sorting operation
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
			d_prevFitnessKeys + offf, d_currFitnessKeys + offf, d_prevFitnessValues + offf, d_currFitnessValues + offf, p);
	}
}

// FINALIZANDO... TESTAR COM MULTIPLAS POPULAÇÕES
template< class Decoder >
std::vector<std::vector<Individual>> BRKGA< Decoder >::getPopulations() const { 
	/*
	float* aux_pop = h_populations;
	float* aux_fitk = h_fitnessKeys;
	int* aux_fitv = h_fitnessKeys;
	
	cpyHost();
	std::vector<std::vector<Individual>> pops; // TESTAR
	for(int _k = 0; _k < K; _k++)
	{
		int offp = getOffset(_k, true);
		int offn = getOffset(_k, false);
		aux_pop = &h_populations[offp];
		aux_fitk = &h_fitnessKeys[offn];
		aux_fitv = &h_fitnessValues[offn];
		for(int _p = 0; _p < p; _p++)
		{
			//INDEX INDIVIDUALS
			//CREATE ONE
			for(int _n = 0; _n < n; _n++)
			{
				// CHROMOSOMES
				// FILL
			}
		}
	}		
	


	return pops;
	*/
}

template< class Decoder >
float* BRKGA< Decoder >::getFitnessKeys() const { return h_fitnessKeys; }

template< class Decoder >
int* BRKGA< Decoder >::getFitnessValues() const { return h_fitnessValues; }

template< class Decoder >
unsigned BRKGA< Decoder >::getN() const { return n; }

template< class Decoder >
unsigned BRKGA< Decoder >::getP() const { return p; }

template< class Decoder >
unsigned BRKGA< Decoder >::getPe() const { return pe; }

template< class Decoder >
unsigned BRKGA< Decoder >::getPm() const { return pm; }

template< class Decoder >
unsigned BRKGA< Decoder >::getPo() const { return p - pe - pm; }

template< class Decoder >
double BRKGA< Decoder >::getRhoe() const { return rhoe; }

template< class Decoder >
unsigned BRKGA< Decoder >::getK() const { return K; }

template< class Decoder >
unsigned BRKGA< Decoder >::getOffset(int _k, bool _pop) const { 
	if(_pop)return _k*p*n; // true, return chromosomes of population k offset
	else return _k*p; // else return fitness of population k offset
}

template< class Decoder > // get gpu pointer to chromosome i of population k in current
unsigned BRKGA< Decoder >::getChromosome(unsigned _k, unsigned i) const { 
	return _k*p*n + i*n;
}

#endif