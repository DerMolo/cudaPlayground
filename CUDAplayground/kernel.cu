
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>


float randomFloat()
{ return (float)rand() / (float)RAND_MAX; } //where x = rand() 
//essentially extracts a random value from a scope RAND_MIN < x < RAND_MAX and determines x's percent value (ranging from 0 to 1) 


template <typename T> 
__global__ void addKernel(T *c, const T *a, const T *b, const unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; /*
    blockDim.x = threadSize of a block in the x dimension 
    blockId.x = block of the current thread 
   */
    if(i < size )
        c[i] = a[i] + b[i];
}

template <typename Ta> 
void addWithCuda(Ta* c, Ta* a, Ta* b, const unsigned int size)
{
    Ta* devA = 0;
    Ta* devB = 0;
    Ta* devC = 0;

     cudaMalloc((void**)&devC, size * sizeof(Ta));
     cudaMalloc((void**)&devA, size * sizeof(Ta));
     cudaMalloc((void**)&devB, size * sizeof(Ta));

     //Syntax notes: 
     /* 
     1) devPtr : pointer to the location of the allocated memory (stored on the CPU
      2) memSize: size of the allocated memory in bytes 

        cudaMalloc description: 
        stores address of the device's allocated memory into devPtr.
        CPU-to-GPU pointer references (without invoking cudMemcpy) will result in a crash.                          
                                                                                                                    
        the function enables data exchange between the GPU and CPU                                                  
                                                                                                                    
     */                                                                                                             
                                                                                                                    
    // Copy input vectors from host memory to GPU buffers.                                                          
     //devA and devB store address of allocated memory in the GPU                                                   
     //array of values are sent to corresponding address                                                            
                                                                                                                    
     cudaMemcpy(devA, a, size * sizeof(Ta), cudaMemcpyHostToDevice);                                                
     cudaMemcpy(devB, b, size * sizeof(Ta), cudaMemcpyHostToDevice);                                                
                                                                                                                    
                                                                                                                    
    // Launch a kernel on the GPU with one thread for each element.                                                 
     int blocks = 1;                                                                                                
     int threadSize = size; 
     if (threadSize > 1024) {
         blocks = ceil((float)size / 1024);
         threadSize = 1024;
         std::cout << "arrSize exceeds per-block thread size\nBlocks calculated: " << blocks << std::endl;
     }
    addKernel<<<blocks, threadSize>>>(devC, devA, devB, size);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
     cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
     cudaMemcpy(c, devC, size * sizeof(Ta), cudaMemcpyDeviceToHost);

}

template <typename T>
void printArr(int start,int end, T arr[]) {
    for (int i = start; i < end; i++)
        std::cout << arr[i] << ", ";
    std::cout<<std::endl;
} 

int main()
{
    std::cout << "max per-grid thread size: " << pow(2, 31) << std::endl;
    unsigned int arraySize = 3000;
    std::cout << "assign arraySize: " << std::endl;
    std::cin >> arraySize;

    //initializes random "vectors" 
     float* a = new float[arraySize];
     float* b= new float[arraySize];
     for (int i = 0; i < arraySize; i++) {
         a[i] = randomFloat(); 
         b[i] = randomFloat(); 
     }

    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };

    //float c[arraySize] = { 0 };
    float* c = new float[arraySize];
    //c = {0};
    // Add vectors in parallel.
    //cudaError_t  = 
   addWithCuda(c, a, b, arraySize);

   std::cout << "printing 20 final elements of passed arrays" << std::endl;
   printArr(arraySize - 20, arraySize, a);
   std::cout << "+ \n";
   printArr(arraySize - 20, arraySize, b);
   std::cout << "= \n";
   printArr(arraySize - 20, arraySize, c);
   std::cout << std::endl;

    cudaDeviceReset();
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

