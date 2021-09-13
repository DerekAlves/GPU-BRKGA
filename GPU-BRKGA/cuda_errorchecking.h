#ifndef CUDA_ERRORCHECKING_H_
#define CUDA_ERRORCHECKING_H_

#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#endif