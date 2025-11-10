
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cooperative_groups.h>

float randomFloat()
{
    return (float)rand() / (float)RAND_MAX;
} //where x = rand() 
//extracts a random value from a scope RAND_MIN < x < RAND_MAX and determines x's percent value (ranging from 0 to 1) 

template <typename T>
__global__ void reverseKernel(T* arr, const unsigned int size) {
    //auto grid = cooperative_groups::this_grid();
    __shared__ bool syncBlock[4];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        T temp = arr[(size - 1) - i];

        int targetBlock = ((size - 1) - i) / 1024;
        __syncthreads();
        syncBlock[blockIdx.x] = true;
        int wait = 0;
        while (wait == 10000) { wait++; } //momentary block for synchronization (across all threads instead of a block) 
        arr[i] = temp;
    }
}

template <typename T>
__global__ void addKernel(T* c, const T* a, const T* b, const unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; /*
    blockDim.x = threadSize of a block in the x dimension
    blockId.x = block of the current thread
   */
    if (i < size)
        c[i] = a[i] + b[i];
}

template <typename T>
__global__ void searchKernel(T* arr, T* target, bool* output, const unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    if (i < size) {
        if (arr[i] == *target) {
            //printf("target found at index %d ", i);
            int wait = 0;
            while (wait == 10000) { wait++; }
            *output = true;
        }
    }
}

void calcBlockAndThreads(int& threadSize, int& blocks) {
    if (threadSize > 1024) {
        blocks = ceil((float)threadSize / 1024);
        threadSize = 1024;
        std::cout << "arrSize exceeds per-block thread size\nBlocks calculated: " << blocks << std::endl;
    }
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
void reverseWithCuda(T* arr, const unsigned int size) {
    T* devPtr = 0;
    cudaMalloc((void**)&devPtr, sizeof(T) * size);

    int blocks = 1;
    int threads = size;
    calcBlockAndThreads(threads, blocks);


    cudaMemcpy(devPtr, arr, sizeof(T) * size, cudaMemcpyHostToDevice);

    /*
    * only functions with cuda 13>
    //boilerplate for cooperative launch
    void* kernelArgs[] = { &devPtr, &size};
    cudaLaunchCooperativeKernel((void*)reverseKernel, 1, blocks, kernelArgs);
    */
    reverseKernel << <blocks, threadSize >> > (devPtr, size);

    cudaDeviceSynchronize();

    cudaMemcpy(arr, devPtr, sizeof(T) * size, cudaMemcpyDeviceToHost);
}


template <typename T>
bool searchWithCuda(T* arr, T* target, const unsigned int size) {
    T* devArr = 0;
    bool* devOut = 0;
    T* devTarget = 0;

    bool* out = new bool; 

    cudaMalloc((void**)&devTarget, sizeof(T));
    cudaMalloc((void**)&devArr, sizeof(T) * size);
    cudaMalloc((void**)&devOut, sizeof(bool));
    int blocks = 1;
    int threads = size;
    calcBlockAndThreads(threads, blocks);

    cudaMemcpy(devTarget, target, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(devArr, arr, sizeof(T) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devOut, out, sizeof(bool), cudaMemcpyHostToDevice);
    //cudaMemcpy(devOut, devOut, sizeof(bool), cudaMemcpyHostToDevice);

    searchKernel << <blocks, threads >> > (devArr, devTarget, devOut, size);
    cudaMemcpy(out, devOut, sizeof(bool), cudaMemcpyDeviceToHost);
    //cudaMemcpy(devOut, devOut, sizeof(bool), cudaMemcpyDeviceToHost);
    //return *devOut;
    return *out;

    /*
    * shouldnt the host be able to dereference devOut into its corresponding value? 
    * if so, would cudaMemcpy be necessary? 
    * testing with: 
    * cudaMemcpy(devOut, devOut, sizeof(bool), cudaMemcpyDeviceToHost)
    * Clarification: 
    * the host cannot dereference device pointers. 
    * cudaMemcpy simply takes device data (using pointers) and transmitting them to the host (and vice versa) 
    */

    cudaDeviceSynchronize();
}

template <typename T>
void printArr(int start, int end, T arr[]) {
    for (int i = start; i < end; i++)
        std::cout << arr[i] << ", ";
    std::cout << std::endl;
}

int main()
{
    /*
        std::cout << "max per-grid thread size: " << pow(2, 31) << std::endl;
    unsigned int arraySize = 3000;
    std::cout << "assign arraySize: " << std::endl;
    std::cin >> arraySize;

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
<<<<<<< HEAD

=======
>>>>>>> b345681c549254b548bee1cf3b2bee5e459706e2
    // Add vectors in parallel.

   addWithCuda(c, a, b, arraySize);

   printArr(arraySize - 20, arraySize, a);
   std::cout << "+ \n";
   printArr(arraySize - 20, arraySize, b);
   std::cout << "= \n";
   printArr(arraySize - 20, arraySize, c);
   std::cout << std::endl;

    cudaDeviceReset();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
<<<<<<< HEAD

    */
    unsigned int arrSize = 3000;
    std::cout << "assign arraySize: " << std::endl;
    std::cin >> arrSize;

    float* a = new float[arrSize];
    for (int i = 0; i < arrSize; i++) {
        a[i] = randomFloat();
    }
    /*
    printArr(arrSize - 5, arrSize, a);
    printArr(0, 5, a);

    reverseWithCuda(a, arrSize);

    printArr(arrSize - 5, arrSize, a);
    printArr(0, 5, a);
    */
    printf("printing arr\n");
    if (arrSize <= 50)
        printArr(0, arrSize, a);
    else
        printArr(0,50,a);
    float* target = new float;
    *target = randomFloat();
    bool output = searchWithCuda(a, target, arrSize);
    if(output)
        printf("\nTARGET: %f, TARGET FOUND: %d", *target,output);
    else 
        printf("\nTARGET: %f, TARGET NOT FOUND: %d", *target,output);

    cudaDeviceReset();

    return 0;
}
