#include <iostream>
#include <algorithm>
#include "GPU-BRKGA/BRKGA.cuh"
#include <cuda_runtime.h>
#include "SampleDecoder.cuh"


int main(int argc, char* argv[]) {


	int GEN = std::atoi(argv[2]);
	printf("\n");

	const unsigned n = 128;		// size of chromosomes
	const unsigned p = 1024;	// size of population
	const double pe = 0.15625;	// fraction of population to be the elite-set
	const double pm = 0.15625;	// fraction of population to be replaced by mutants
	const double rhoe = 0.70;	// probability that offspring inherit an allele from elite parent
	const unsigned K = 1;		// number of independent populations
	unsigned ipt = 1;			// number of threads for parallel decoding
	unsigned thr = n;
	bool gpu_deco = true;

	SampleDecoder RefDecoder;
	int se = std::atoi(argv[1]);

	float ms;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	// initialize the BRKGA-based heuristic
	BRKGA<SampleDecoder> algorithm(n, p, pe, pm, rhoe, RefDecoder, se, ipt, thr, gpu_deco, K);

	unsigned generation = 0;		// current generation
	const unsigned MAX_GENS = GEN;	// run for 1000 gens
	//std::cout << "Running for " << MAX_GENS << " generations..." << std::endl;
	do {
		algorithm.evolve();	// evolve the population for one generation
	}while(generation++ < MAX_GENS);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	

	Individual ind = algorithm.getBestIndividual();
	printf("Fitness: %.3f ", ind.fitness.first);

	printf(" Time: %9.3f milliseconds\n", ms);
	return 0;
}