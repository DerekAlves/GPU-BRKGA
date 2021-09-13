/* api-usage.cu
This file should be specificated by the user.

Authors
Derek N.A. Alves, 
Bruno C.S. Nogueira, 
Davi R.C Oliveira and
Ermeson C. Andrade.

Instituto de Computação, Universidade Federal de Alagoas.
Maceió, Alagoas, Brasil.
*/
 
#include <iostream>
#include <algorithm>
#include "GPU-BRKGA/GPUBRKGA.cuh"
#include <cuda_runtime.h>
#include "SampleDecoder.cuh"

int main(int argc, char* argv[]) {

	int GEN = std::atoi(argv[2]);// generations from argv
	//printf("\n");
	
	const unsigned n = 32;		// size of chromosomes
	const unsigned p = 256;	// size of population
	const double pe = 0.15625;	// fraction of population to be the elite-set
	const double pm = 0.15625;	// fraction of population to be replaced by mutants
	const double rhoe = 0.70;	// probability that offspring inherit an allele from elite parent
	const unsigned K = 1;		// number of independent populations
	unsigned ipt = 1;	        // number items per threads for parallel evolution
	
	SampleDecoder RefDecoder; 	// reference to decoder
	

	int se = std::atoi(argv[1]);//seed from argv
	
	//time measuring
	float ms;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	// initialize GPU-BRKGA
	GPUBRKGA<SampleDecoder> algorithm(n, p, pe, pm, rhoe, RefDecoder, se, ipt, K);
	
	
	unsigned generation = 0;		// current generation
	const unsigned X_INTVL = 1000;	// exchange best individuals at every 100 generations
	const unsigned X_NUMBER = 2;	// exchange top 2 best
	const unsigned MAX_GENS = GEN;	// run for 1000 gens
	
	//std::cout << "Running for " << MAX_GENS << " generations..." << std::endl;
	do {
		algorithm.evolve(); // evolve the population for one generation
		generation++;
		//printf("GENERATION %d------------------------------\n", generation);
	} while (generation < MAX_GENS);  	

	Individual ind = algorithm.getBestIndividual();
	printf("Fitness: %.3f\n", ind.fitness.first);

	//example get population in device
	std::vector<std::vector<Individual*>> pops = algorithm.getDeviceInfo();

	/*for(int i = 0; i < p; i++)
	{
		//Individual aux = pops[0][i];
		//aux.ToString();
		pops[0][i]->ToString();
	}*/

	//time measuring
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
 
	printf(" Time: %9.3f milliseconds\n", ms);
	return 0;
}