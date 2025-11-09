
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* devA = 0;
    int* devB = 0;
    int* devC = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
     cudaSetDevice(0);

    // Allocate GPU buffers for three vectors (two input, one output)    .
     cudaMalloc((void**)&devC, size * sizeof(int));
     cudaMalloc((void**)&devA, size * sizeof(int));

     cudaMalloc((void**)&devB, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
     cudaMemcpy(devA, a, size * sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(devB, b, size * sizeof(int), cudaMemcpyHostToDevice);


    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size >>>(devC, devA, devB);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
     cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
     cudaMemcpy(c, devC, size * sizeof(int), cudaMemcpyDeviceToHost);

}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    //cudaError_t  = 
   addWithCuda(c, a, b, arraySize);

    for (int i = 0; i < 5; i++)
        std::cout << a[i] << ", ";
    std::cout << " + ";

    for (int i = 0; i < 5; i++)
        std::cout << b[i] << ", ";
    std::cout << "=  ";

    for (int i = 0; i < 5; i++)
        std::cout << c[i] << ", ";
    std::cout << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*
        = cudaDeviceReset();
    if ( != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    */
 
    return 0;
}

