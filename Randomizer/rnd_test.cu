 //Compiler command-line:
 //$ nvcc -lineinfo rnd_test.cu -o rnd_test
 //Profiler command-line:
 //$ nvvp ./rnd_test

//Includes the mtrand methods
extern "C" {
//Comment out this #define if you want to run on device!
//#define HOST
#include "mtrand.c"
}

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <chrono>

#ifndef HOST
#define CUDA_GLOBAL __global__
//If HOST is not defined, then this code is running on the GPU (device)
#else
#define CUDA_GLOBAL
#endif

//Computes the random vals and stores it in the array
CUDA_GLOBAL void kernel(double *rndVals, int n){
    //Creates rndGen structure
    struct MTrand_Info rndGen;

#ifndef HOST
    //Following the logic from NVIDIA's beginners tutorial
    
    //Index is the thread's offset from the beginning of the block and the thread's index added in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //Initializes MTrand base call and changes seed each time based upon threadId
    MTrand_init(&rndGen, 5489+threadIdx.x);
    //Checks to make sure the index is within the size of the array
    if (index < n) {
        rndVals[index] = MTrand_get(&rndGen);
    }
#else
    for (int i = 0; i < n; i++) {
        //Initialize MTrand base call and changes seed each time based upon index
        MTrand_init(&rndGen, 5489 + i);
        rndVals[i] = MTrand_get(&rndGen);
    }
#endif
}

//Calls the kernel/function
void genRndNums(const int reps = 100) {

    //Create a variable to hold all of the random numbers
    double *rndVals;
    
    //Allocate variable in unified memory
    //(memory available to both CPU and GPU)
    cudaMallocManaged(&rndVals, reps * sizeof(double));

#ifndef HOST
    //Sets up the block size and number of blocks for the kernel

    //Included in order to get the maximum from each streaming multiprocessor
    //inside of GPU
    int blockSize = 1024;
    //Calculates the number of thread blocks in the grid
    int numOfBlocks = (int) ceil( (float) reps / blockSize);

    //Calls the CUDA kernel and generates 'reps' number of random numbers
    kernel<<<numOfBlocks, blockSize>>>(rndVals, reps);
#else
    //Calls the function normally and generates 'reps' number of random numbers
    kernel(rndVals, reps);
#endif

    //Waits for the GPU code to finish executing before re-accessing the host
    //Synchronizes devices
    cudaDeviceSynchronize();
    
    //Loop through rndVals and prints out the results
    for (int i = 0; (i < reps); i++) {
        std::cout << rndVals[i] << " ";
    }
    std::cout << std::endl;
    
    //Free Memory
    cudaFree(rndVals);

}

//Calls getRndNums
//Defaults to 100 reps if arguement is not supplied
int main(int argc, char *argv[]) {
    auto startTime = std::chrono::high_resolution_clock::now();
    const int reps = (argc > 1 ? std::stoi( argv[1] ) : 100);
    genRndNums(reps);
    std::cout << "Success, " << reps << " random numbers created!" << std::endl;
#ifdef HOST
    std::cout << "Compiled on HOST (CPU)" << std::endl;
#else
    std::cout << "Compiled on DEVICE (GPU)" << std::endl;
#endif
    auto stopTime = std::chrono::high_resolution_clock::now();

    //Calculates the time take for the code to run
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime);

    std::cout << "Time Taken: " <<
        time.count() << " microseconds" << std::endl;
    return 0;
}
