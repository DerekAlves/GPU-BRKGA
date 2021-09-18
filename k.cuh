/* k.cuh
Auxiliar file, this file contains the kernel for
GPU decodification, should be specificated by the user.

Authors
Derek N.A. Alves, 
Bruno C.S. Nogueira, 
Davi R.C Oliveira and
Ermeson C. Andrade.

Instituto de Computação, Universidade Federal de Alagoas.
Maceió, Alagoas, Brasil.
*/
 

#ifndef K_CUH
#define K_CUH

__global__ void dec(float* d_next, float* d_nextFitKeys)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float xi = (d_next[idx] - 0.499) * 10.24;
	float value = powf(xi, 2) - 10 * cosf(2 * 3.1416 * xi);

	typedef cub::BlockReduce<float, 32> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float sum = BlockReduce(temp_storage).Reduce(value, cub::Sum());
	
	if(threadIdx.x == 0)
	{
		d_nextFitKeys[blockIdx.x] = 10 * blockDim.x + sum;
	}
				
}

#endif