#include <string>
#include<list>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <cuda.h>
#include<cuda_runtime.h>

    bool isGPUPtr(const void *ptr)
    {
    	cudaPointerAttributes atts;
    	const cudaError_t perr = cudaPointerGetAttributes(&atts, ptr);

    // clear last error so other error checking does
    // not pick it up
    	cudaError_t error = cudaGetLastError();
#if CUDART_VERSION >= 10000
    	return perr == cudaSuccess &&
        	(atts.type == cudaMemoryTypeDevice ||
         	atts.type == cudaMemoryTypeManaged);
#else
    	return perr == cudaSuccess && atts.memoryType == cudaMemoryTypeDevice;
#endif
    }
