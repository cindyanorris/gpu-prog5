#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "config.h" //defines the number of threads per block and the number of blocks
#include "d_vecScalarMult.h"

//put the prototypes for the two kernels here

/*  d_vecScalarMult
    Multiples a vector by a scalar using the GPU (the device).
    A is a pointer to the input vector.
    K is the scalar to use in the multiply.
    The result is stored in the vector pointed to by R.
    n is the length of the vectors.
    which indicates whether the work should be distributed among the threads
    using BLOCK partitioning or CYCLIC partitioning.

    returns the amount of time it takes to perform the
    vector scalar multiply
*/
float d_vecScalarMult(float* A, float * R, float K, int n, int which)
{
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;

    //time the sum
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

    //your code to prepare for and invoke the kernel goes here

    /*
    if (which == BLOCK)
       //call kernel that uses block partitioning  
    else
       //call kernel that uses cyclic partitioning  
    */


    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
    return gpuMsecTime;
}

//put the two kernels here
