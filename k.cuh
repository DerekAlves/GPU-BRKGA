#ifndef K_CUH
#define K_CUH

//Kernel for GPU Decodification
__global__ void dec(float* d_next, float* d_nextFitKeys, int* d_nextFitValues)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float xi = (d_next[idx] - 0.5) * 10.24;
	float value = powf(xi, 2) - 10 * cosf(2 * 3.1416 * xi);

	typedef cub::BlockReduce<float, 128> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float sum = BlockReduce(temp_storage).Reduce(value, cub::Sum());
	
	if(!threadIdx.x)
		d_nextFitKeys[blockIdx.x] = 10 * blockDim.x + sum;
				
}

#endif